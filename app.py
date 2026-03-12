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
from db_models.user import User, GlobalSetting
from functools import wraps
from utils.gemini_analyzer import GeminiAnalyzer

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

db.init_app(app)
bcrypt.init_app(app)
login_manager.init_app(app)

from db_models.user import User

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

# Note: Models are now lazy-loaded by model_manager
def load_models():
    return model_manager.load_models()

# التأكد من وجود المجلدات المطلوبة مسبقاً
with app.app_context():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DATA_RAW_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DATA_PROCESSED_FOLDER'], exist_ok=True)

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
        if 'file' not in request.files:
            return jsonify({'error': 'لم يتم رفع أي ملف'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'الملف فارغ'}), 400
        
        if file and allowed_file(file.filename):
            original_filename = file.filename
            file_ext = os.path.splitext(original_filename)[1].lower()
            # Add timestamp to name to avoid collisions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{secure_filename_arabic(original_filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if file_ext == ".csv":
                df = pd.read_csv(filepath)
            elif file_ext == ".xlsx":
                df = pd.read_excel(filepath, engine="openpyxl")
            elif file_ext == ".xls":
                df = pd.read_excel(filepath, engine="xlrd")
            else:
                os.remove(filepath)
                return jsonify({"error": f"Unsupported file format: {file_ext}"}), 400
            
            # Check limits for guests
            if not current_user.is_authenticated:
                limit_setting = GlobalSetting.query.filter_by(key='guest_analysis_limit').first()
                limit = int(limit_setting.value) if limit_setting else 10
                if len(df) > limit:
                    os.remove(filepath)
                    return jsonify({'error': f'عذراً، كضيف يمكنك تحليل {limit} أسطر فقط. يرجى تسجيل الدخول للتحليل غير المحدود.'}), 400
            
            raw_path = os.path.join(app.config['DATA_RAW_FOLDER'], f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{filename}')
            df.to_csv(raw_path, index=False)
            
            return jsonify({'success': True, 'filename': filename, 'rows': len(df), 'columns': list(df.columns)})
        
        return jsonify({'error': 'نوع الملف غير مسموح'}), 400
    return render_template('upload.html')

@app.route('/analyze/<filename>', methods=['POST'])
def analyze_file(filename):
    base_name = os.path.splitext(filename)[0]
    processed_path = os.path.join(app.config['DATA_PROCESSED_FOLDER'], f'analyzed_{base_name}.csv')
    summary_path = os.path.join(app.config['DATA_PROCESSED_FOLDER'], f'analyzed_{base_name}.json')
    
    # Use Cache if exists
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
            return jsonify({'success': True, 'results': results, 'stats': _calculate_stats(results)})
        except Exception as e:
            print(f"Cache Reading Error: {e}")
            # If cache is corrupt, proceed to re-analyze

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'الملف غير موجود'}), 404
        
    if filename.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    
    text_column = next((col for col in ['text', 'review', 'comment', 'نص', 'تقييم'] if col in df.columns), None)
    if text_column is None:
        return jsonify({'error': 'لم يتم العثور على عمود النص في الملف'}), 400
    
    texts = df[text_column].fillna("").astype(str).tolist()
    
    force_local = request.args.get('force_local') == 'true'
    provider = 'local'
    
    # Analyze using Gemini or Local Model
    if current_user.is_authenticated and current_user.use_gemini and current_user.gemini_api_key and not force_local:
        try:
            analyzer = GeminiAnalyzer(current_user.gemini_api_key)
            results = analyzer.analyze_batch(texts)
            provider = 'gemini'
        except Exception as e:
            err_msg = str(e)
            print(f"Gemini Error: {err_msg}")
            # Identify if it's a provider error that needs interactive fallback
            if any(k in err_msg.lower() for k in ["quota", "api_key", "limit"]):
                return jsonify({
                    'success': False, 
                    'error_type': 'provider_error', 
                    'message': f"فشل الاتصال بـ Gemini: {err_msg}",
                    'details': err_msg
                }), 400
            
            # Other errors fallback automatically
            load_models()
            results = model_manager.analyze_batch(texts)
    else:
        try:
            load_models()
            results = model_manager.analyze_batch(texts)
        except Exception as e:
            print(f"Local Analysis Crash: {e}")
            return jsonify({'error': f'فشل التحليل المحلي: {str(e)}'}), 500
    
    # Ensure provider and logic_explanation are marked in each result
    for r in results:
        if 'provider' not in r:
            r['provider'] = provider
        if 'logic_explanation' not in r:
            r['logic_explanation'] = 'تحليل محلي عبر موديل BERT' if provider == 'local' else ''

    results_df = pd.DataFrame(results)
    results_df.to_csv(processed_path, index=False)
    
    # Generate and save Executive Summary if Gemini is configured (even if analysis fell back to local)
    if current_user.is_authenticated and current_user.use_gemini and current_user.gemini_api_key:
        try:
            temp_analyzer = GeminiAnalyzer(current_user.gemini_api_key)
            summary_text = temp_analyzer.generate_executive_summary(results)
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({'executive_summary': summary_text}, f, ensure_ascii=False)
        except Exception as e:
            print(f"Executive Summary Generation failed: {e}")
    
    return jsonify({'success': True, 'results': results, 'stats': _calculate_stats(results)})

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        current_user.use_gemini = 'use_gemini' in request.form
        current_user.gemini_api_key = request.form.get('gemini_api_key')
        db.session.commit()
        flash('تمت تحديث الإعدادات بنجاح', 'success')
    return render_template('settings.html')

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
    return {
        'total_reviews': len(results),
        'sentiment_distribution': {
            'إيجابي': sum(1 for r in results if r.get('overall_sentiment') == 'إيجابي'),
            'سلبي': sum(1 for r in results if r.get('overall_sentiment') == 'سلبي'),
            'محايد': sum(1 for r in results if r.get('overall_sentiment', 'محايد') == 'محايد')
        }
    }

@app.route('/results/<filename>')
def get_results(filename):
    """جلب النتائج المحللة مسبقاً للداش بورد"""
    return analyze_file(filename)

@app.route('/dashboard/<filename>')
@login_required
def dashboard(filename):
    return render_template('dashboard.html', filename=filename)

@app.route('/summary/<filename>')
@login_required
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
    
    return send_file(filepath, as_attachment=True)


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