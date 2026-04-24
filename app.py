from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from flask_login import login_user, current_user, logout_user, login_required
from flask_cors import CORS
import pandas as pd
import os
import json
from werkzeug.utils import secure_filename
import plotly
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import ast
import markdown

from config import Config
from extensions import db, bcrypt, login_manager
from db_models.user import User, GlobalSetting, AnalysisHistory
from functools import wraps
from utils.gemini_analyzer import GeminiAnalyzer
from utils.reporting_service import AnalysisReportingService
from utils.upload_service import build_analysis_cache_key, detect_text_column, read_uploaded_dataframe

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

db.init_app(app)
bcrypt.init_app(app)
login_manager.init_app(app)

from db_models.user import User

def initialize_database():
    """Ensure required tables and defaults exist before serving requests."""
    with app.app_context():
        db.create_all()

        if not GlobalSetting.query.filter_by(key='guest_analysis_limit').first():
            db.session.add(GlobalSetting(key='guest_analysis_limit', value='50'))

        admin_email = "admin@arasent.ai"
        if not User.query.filter_by(email=admin_email).first():
            hashed_pw = bcrypt.generate_password_hash('admin123').decode('utf-8')
            db.session.add(User(
                username='admin',
                email=admin_email,
                password_hash=hashed_pw,
                is_admin=True
            ))

        db.session.commit()

initialize_database()

@login_manager.user_loader
def load_user(user_id):
    if not user_id:
        return None
    try:
        return db.session.get(User, int(user_id))
    except:
        return None

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('غير مسموح! تحتاج إلى صلاحيات المسؤول', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

from utils.manager import model_manager

def ensure_storage_dirs():
    """Create storage directories if they are missing at runtime."""
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DATA_RAW_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DATA_PROCESSED_FOLDER'], exist_ok=True)

# التأكد من وجود المجلدات المطلوبة مسبقاً
with app.app_context():
    ensure_storage_dirs()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def secure_filename_arabic(filename):
    """
    نسخة من secure_filename تدعم الحروف العربية
    تقوم بتنظيف اسم الملف من الرموز الخطرة مع الحفاظ على الحروف والارقام (بما فيها العربية)
    """
    # فصل الامتداد
    name, ext = os.path.splitext(filename)
    
    # تنظيف الاسم: السماح بالحروف والأرقام (تشمل العربية في بايثون 3) والنقاط والشرطات والمسافات
    clean_name = re.sub(r'[^\w\.\-\s]', '', name)
    
    # استبدال المسافات بشرطة سفلية
    clean_name = re.sub(r'\s+', '_', clean_name)
    
    # إزالة النقاط والشرطات من البداية والنهاية
    clean_name = clean_name.strip('._-')
    
    if not clean_name:
        clean_name = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    return f"{clean_name}{ext.lower()}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_exists = User.query.filter((User.username == username) | (User.email == email)).first()
        if user_exists:
            flash('اسم المستخدم أو البريد الإلكتروني موجود بالفعل', 'danger')
            return render_template('register.html')
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password_hash=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('تم إنشاء حسابك بنجاح! يمكنك الآن تسجيل الدخول', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user, remember=request.form.get('remember'))
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('فشل تسجيل الدخول. يرجى التحقق من البريد الإلكتروني وكلمة المرور', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        ensure_storage_dirs()

        if 'file' not in request.files:
            return jsonify({'error': 'لم يتم رفع أي ملف'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'الملف فارغ'}), 400
        
        if file and allowed_file(file.filename):
            original_filename = file.filename
            # Add timestamp to name to avoid collisions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{secure_filename_arabic(original_filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                df = read_uploaded_dataframe(filepath)
            except Exception as e:
                os.remove(filepath)
                return jsonify({"error": f"تعذر قراءة الملف المرفوع: {str(e)}"}), 400
            
            # Check limits for guests
            is_sliced = False
            slice_message = ""
            if not current_user.is_authenticated:
                limit_setting = GlobalSetting.query.filter_by(key='guest_analysis_limit').first()
                limit = int(limit_setting.value) if limit_setting else 50
                if len(df) > limit:
                    df = df.head(limit)
                    is_sliced = True
                    slice_message = f"تم تحليل أول {limit} تقييمات كعينة تجريبية. لمعاينة الملف كاملاً، تفضل بإنشاء حساب مجاني!"
            
            column_detection = detect_text_column(df)

            raw_path = os.path.join(app.config['DATA_RAW_FOLDER'], f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{filename}')
            df.to_csv(raw_path, index=False)
            
            return jsonify({
                'success': True, 
                'filename': filename, 
                'rows': len(df), 
                'columns': list(df.columns),
                'column_detection': column_detection,
                'sliced': is_sliced,
                'slice_message': slice_message
            })
        
        return jsonify({'error': 'نوع الملف غير مسموح'}), 400
    return render_template('upload.html')

@app.route('/analyze/<filename>', methods=['POST'])
def analyze_file(filename):
    ensure_storage_dirs()

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'الملف غير موجود'}), 404
        
    try:
        df = read_uploaded_dataframe(filepath)
    except Exception as e:
        return jsonify({'error': f'تعذر قراءة الملف: {str(e)}'}), 400

    requested_column = request.args.get('text_column', '').strip()
    if requested_column and requested_column in df.columns:
        text_column = requested_column
    else:
        detection = detect_text_column(df)
        text_column = detection['selected']

    if text_column is None:
        return jsonify({
            'error': 'لم نتمكن من تحديد عمود التعليقات تلقائيًا. اختر العمود المناسب قبل بدء التحليل.',
            'columns': list(df.columns),
            'column_detection': detect_text_column(df),
        }), 400
    
    # === تنظيف البيانات قبل التحليل ===
    df = df.dropna(subset=[text_column])
    df = df.drop_duplicates(subset=[text_column])
    texts = df[text_column].astype(str).str.replace('\n', ' ', regex=False).str.replace('\r', ' ', regex=False).str.strip().tolist()
    texts = [t for t in texts if t]
    
    if not texts:
        return jsonify({'error': 'الملف لا يحتوي على أي نصوص صالحة للتحليل بعد التنظيف'}), 400
    
    force_arasant = request.args.get('force_arasant') == 'true'
    # اختيار الموديل من المستخدم: arasant | gemini
    selected_model = request.args.get('model', 'arasant')
    provider = selected_model

    # === نظام التخزين المؤقت الذكي (Smart Cashing) ===
    # ندمج كل النصوص مع اسم الموديل المختار ليكون البصمة فريدة لكل ملف+موديل
    file_hash = build_analysis_cache_key(texts, selected_model)
    
    # الـ base_name الجديد لكل العمليات هو الـ Hash 
    base_name = file_hash
    processed_path = os.path.join(app.config['DATA_PROCESSED_FOLDER'], f'analyzed_{base_name}.csv')
    summary_path = os.path.join(app.config['DATA_PROCESSED_FOLDER'], f'analyzed_{base_name}.json')
    
    # === إذا كانت النتيجة موجودة مسبقاً (Cache Hit) ===
    if os.path.exists(processed_path):
        try:
            print(f" Cache Hit! Returning previous analysis for model {selected_model}")
            df_cached = pd.read_csv(processed_path)
            results = []
            for _, row in df_cached.iterrows():
                try:
                    aspects_raw = row.get('aspects', '[]')
                    aspects = ast.literal_eval(aspects_raw) if isinstance(aspects_raw, str) else aspects_raw
                except:
                    aspects = []
                
                results.append({
                    'original_text': row.get('original_text', ''),
                    'cleaned_text': row.get('cleaned_text', ''),
                    'aspects': aspects,
                    'overall_sentiment': row.get('overall_sentiment', 'محايد'),
                    'confidence': float(row.get('confidence', 0.9)),
                    'logic_explanation': row.get('logic_explanation', ''),
                    'provider': row.get('provider', 'cached')
                })
            stats = _calculate_stats(results)
            
            # تسجيل العملية في History كـ (إعادة اطلاع) بدون إعادة تحليل
            if current_user.is_authenticated:
                try:
                    # internal_filename is globally unique in the current schema.
                    existing_record = AnalysisHistory.query.filter_by(
                        internal_filename=base_name
                    ).first()

                    if not existing_record:
                        history_record = AnalysisHistory(
                            user_id=current_user.id,
                            original_filename=os.path.splitext(filename)[0],
                            internal_filename=base_name,
                            model_used=provider,
                            total_reviews=stats['total_reviews'],
                            positive_count=stats['sentiment_distribution']['إيجابي'],
                            negative_count=stats['sentiment_distribution']['سلبي'],
                            neutral_count=stats['sentiment_distribution']['محايد']
                        )
                        db.session.add(history_record)
                        db.session.commit()
                except Exception as history_error:
                    db.session.rollback()
                    print(f"History save skipped on cache-hit: {history_error}")
                
            return jsonify({
                'success': True,
                'results': results,
                'stats': stats,
                'cached': True,
                'base_name': base_name,
                'text_column': text_column,
            })
        except Exception as e:
            db.session.rollback()
            print(f"Cache Reading Error: {e}, proceeding to re-analyze")
            # If cache is corrupt, proceed to re-analyze

    # ========================================================
    # إذا لم يكن موجوداً (Cache Miss) -> نبدأ التحليل الفعلي
    # ========================================================
    results = []

    # مسار 1: الموديل متعدد المهام الجديد
    if selected_model == 'arasant':
        try:
            from utils.manager import model_manager
            if not getattr(model_manager, 'arasant_analyzer', None):
                if not model_manager.load_arasant_model():
                    return jsonify({'error': 'ملفات موديل AraSent غير موجودة في الخادم. يرجى التأكد من رفع مجلد الموديل.'}), 500
            
            if getattr(model_manager, 'arasant_analyzer', None) and model_manager.arasant_analyzer.model is not None:
                results = model_manager.analyze_batch_arasant(texts)
                provider = 'arasant_local'
            else:
                return jsonify({'error': 'ملفات موديل AraSent غير موجودة أو تالفة.'}), 500
        except Exception as e:
            print(f"AraSent Model Error: {e}")
            return jsonify({'error': f'حدث خطأ أثناء تحميل موديل AraSent: {str(e)}'}), 500

    # مسار 2: Gemini
    elif selected_model == 'gemini' and current_user.is_authenticated and current_user.use_gemini and current_user.gemini_api_key:
        try:
            temp_analyzer = GeminiAnalyzer(current_user.gemini_api_key)
            results = temp_analyzer.analyze_batch(texts)
            provider = 'gemini'
        except Exception as e:
            print(f"Gemini Analysis Error: {e}")
            return jsonify({'error': f'فشل الاتصال بـ Gemini المحترف: {str(e)}'}), 500

    # مسار 3: AraSent كمسار افتراضي وFallback
    else:
        try:
            if not getattr(model_manager, 'arasant_analyzer', None):
                if not model_manager.load_arasant_model(force=force_arasant):
                    return jsonify({'error': 'ملفات موديل AraSent غير موجودة في الخادم. يرجى التأكد من رفع الموديل.'}), 500

            if getattr(model_manager, 'arasant_analyzer', None) and model_manager.arasant_analyzer.model is not None:
                results = model_manager.analyze_batch_arasant(texts)
                provider = 'arasant_local'
            else:
                return jsonify({'error': 'موديل AraSent غير متاح أو تالف.'}), 500
        except Exception as e:
            print(f"AraSent Fallback Error: {e}")
            return jsonify({'error': f'حدث خطأ أثناء تشغيل موديل AraSent: {str(e)}'}), 500
        
    # حفظ النتائج في CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(processed_path, index=False)
    
    # ملخص تنفيذي ديناميكي
    if provider == 'gemini' and current_user.is_authenticated and current_user.use_gemini and current_user.gemini_api_key:
        try:
            temp_analyzer = GeminiAnalyzer(current_user.gemini_api_key)
            summary_text = temp_analyzer.generate_executive_summary(results)
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({'executive_summary': summary_text, 'summary_source': 'gemini'}, f, ensure_ascii=False)
        except Exception as e:
            print(f"Executive Summary Generation failed: {e}")
            summary_text = AnalysisReportingService.generate_local_executive_summary(results)
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({'executive_summary': summary_text, 'summary_source': 'arasant'}, f, ensure_ascii=False)
    else:
        summary_text = AnalysisReportingService.generate_local_executive_summary(results)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({'executive_summary': summary_text, 'summary_source': 'arasant'}, f, ensure_ascii=False)
            
    stats = _calculate_stats(results)
    
    # === حفظ السجل في قاعدة البيانات (للمسجلين فقط) ===
    if current_user.is_authenticated:
        try:
            # Check if this exact analysis already exists to avoid duplicates on refresh
            existing_record = AnalysisHistory.query.filter_by(
                internal_filename=base_name
            ).first()
            
            if not existing_record:
                history_record = AnalysisHistory(
                    user_id=current_user.id,
                    original_filename=os.path.splitext(filename)[0],
                    internal_filename=base_name,
                    model_used=provider,
                    total_reviews=stats['total_reviews'],
                    positive_count=stats['sentiment_distribution']['إيجابي'],
                    negative_count=stats['sentiment_distribution']['سلبي'],
                    neutral_count=stats['sentiment_distribution']['محايد']
                )
                db.session.add(history_record)
                db.session.commit()
                print(f"Saved AnalysisHistory for {base_name}")
        except Exception as e:
            print(f"Error saving analysis history: {e}")
            db.session.rollback()
    
    return jsonify({
        'success': True,
        'results': results,
        'stats': stats,
        'base_name': base_name,
        'text_column': text_column,
    })

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        current_user.use_gemini = 'use_gemini' in request.form
        current_user.gemini_api_key = request.form.get('gemini_api_key')
        db.session.commit()
        flash('تمت تحديث الإعدادات بنجاح', 'success')
    return render_template('settings.html')

@app.route('/history')
@login_required
def history():
    analyses = AnalysisHistory.query.filter_by(user_id=current_user.id).order_by(AnalysisHistory.created_at.desc()).all()
    return render_template('history.html', analyses=analyses)

@app.route('/history/delete/<int:analysis_id>', methods=['POST'])
@login_required
def delete_history(analysis_id):
    analysis = AnalysisHistory.query.get_or_404(analysis_id)
    if analysis.user_id != current_user.id:
        return jsonify({'success': False, 'error': 'غير مصرح'}), 403
        
    try:
        # Delete associated files if they exist
        processed_path = os.path.join(app.config['DATA_PROCESSED_FOLDER'], f'analyzed_{analysis.internal_filename}.csv')
        summary_path = os.path.join(app.config['DATA_PROCESSED_FOLDER'], f'analyzed_{analysis.internal_filename}.json')
        
        if os.path.exists(processed_path):
            os.remove(processed_path)
        if os.path.exists(summary_path):
            os.remove(summary_path)
            
        db.session.delete(analysis)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting history: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/admin', methods=['GET', 'POST'])
@admin_required
def admin():
    limit_setting = GlobalSetting.query.filter_by(key='guest_analysis_limit').first()
    if request.method == 'POST':
        limit_setting.value = request.form.get('guest_analysis_limit')
        db.session.commit()
        flash('تم تحديث الإعدادات العامة', 'success')
    return render_template('admin.html', guest_limit=limit_setting.value)

def _calculate_stats(results):
    return AnalysisReportingService.calculate_stats(results)

@app.route('/results/<base_name>')
def get_results(base_name):
    """جلب النتائج المحللة مسبقاً للداش بورد"""
    base_name = os.path.splitext(base_name)[0]
    processed_path = os.path.join(app.config['DATA_PROCESSED_FOLDER'], f'analyzed_{base_name}.csv')
    
    if os.path.exists(processed_path):
        try:
            df_cached = pd.read_csv(processed_path)
            results = []
            for _, row in df_cached.iterrows():
                try:
                    aspects_raw = row.get('aspects', '[]')
                    aspects = ast.literal_eval(aspects_raw) if isinstance(aspects_raw, str) else aspects_raw
                except:
                    aspects = []
                
                results.append({
                    'original_text': row.get('original_text', ''),
                    'cleaned_text': row.get('cleaned_text', ''),
                    'aspects': aspects,
                    'overall_sentiment': row.get('overall_sentiment', 'محايد'),
                    'confidence': float(row.get('confidence', 0.9)),
                    'logic_explanation': row.get('logic_explanation', ''),
                    'provider': row.get('provider', 'cached')
                })
            stats = _calculate_stats(results)
            return jsonify({'success': True, 'results': results, 'stats': stats})
        except Exception as e:
            return jsonify({'success': False, 'error': f"Error reading cached results: {e}"}), 500
    
    return jsonify({'success': False, 'error': 'Analysis results not found.'}), 404

@app.route('/dashboard/<filename>')
def dashboard(filename):
    return render_template('dashboard.html', filename=filename)

@app.route('/summary/<filename>')
def summary(filename):
    base_name = os.path.splitext(filename)[0]
    filepath = os.path.join(app.config['DATA_PROCESSED_FOLDER'], f'analyzed_{base_name}.csv')
    summary_path = os.path.join(app.config['DATA_PROCESSED_FOLDER'], f'analyzed_{base_name}.json')
    
    if not os.path.exists(filepath):
        return render_template('summary.html', error='المسار غير موجود: ' + filepath)
    
    df = pd.read_csv(filepath)
    
    total_reviews = len(df)
    if 'overall_sentiment' in df.columns:
        positive_count = len(df[df['overall_sentiment'] == 'إيجابي'])
        negative_count = len(df[df['overall_sentiment'] == 'سلبي'])
        neutral_count = len(df[df['overall_sentiment'] == 'محايد'])
    else:
        positive_count = negative_count = neutral_count = 0
    
    positive_pct = (positive_count / total_reviews * 100) if total_reviews > 0 else 0
    negative_pct = (negative_count / total_reviews * 100) if total_reviews > 0 else 0
    neutral_pct = (neutral_count / total_reviews * 100) if total_reviews > 0 else 0
    
    # Load Executive Summary if exists
    executive_summary = None
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            executive_summary = summary_data.get('executive_summary')
            if executive_summary:
                executive_summary = markdown.markdown(executive_summary)
    
    return render_template('summary.html',
                         filename=filename,
                         total_reviews=total_reviews,
                         positive_count=positive_count,
                         negative_count=negative_count,
                         neutral_count=neutral_count,
                         positive_pct=round(positive_pct, 1),
                         negative_pct=round(negative_pct, 1),
                         neutral_pct=round(neutral_pct, 1),
                         executive_summary=executive_summary)

@app.route('/download/<filename>')
def download_file(filename):
    base_name = os.path.splitext(filename)[0]
    filepath = os.path.join(app.config['DATA_PROCESSED_FOLDER'], f'analyzed_{base_name}.csv')
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'الملف غير موجود'}), 404

    df_cached = pd.read_csv(filepath)
    results = []
    for _, row in df_cached.iterrows():
        try:
            aspects_raw = row.get('aspects', '[]')
            aspects = ast.literal_eval(aspects_raw) if isinstance(aspects_raw, str) else aspects_raw
        except Exception:
            aspects = []

        results.append({
            'original_text': row.get('original_text', ''),
            'cleaned_text': row.get('cleaned_text', ''),
            'aspects': aspects,
            'overall_sentiment': row.get('overall_sentiment', 'محايد'),
            'confidence': float(row.get('confidence', 0.0) or 0.0),
            'logic_explanation': row.get('logic_explanation', ''),
            'provider': row.get('provider', ''),
        })

    summary_path = os.path.join(app.config['DATA_PROCESSED_FOLDER'], f'analyzed_{base_name}.json')
    summary_text = AnalysisReportingService.load_summary_text(summary_path, results)
    temp_file_path, download_name = AnalysisReportingService.export_results_to_excel(results, summary_text, base_name)

    return send_file(
        temp_file_path,
        as_attachment=True,
        download_name=download_name,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


@app.route('/about')
def about():
    """
    صفحة "عن المنصة"
    تعرض معلومات عن المنصة ورؤيتها ورسالتها
    """
    return render_template('about.html')

@app.route('/team')
def team():
    """
    صفحة "فريق العمل"
    تعرض أعضاء الفريق ومؤهلاتهم
    """
    return render_template('team.html')  # نفس صفحة about لأنها تحتوي على قسم الفريق

@app.route('/contact')
def contact():
    """
    صفحة "اتصل بنا"
    (يمكن إنشاء هذه الصفحة لاحقاً)
    """
    return render_template('contact.html')  # سننشئ هذه الصفحة عند الحاجة

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
