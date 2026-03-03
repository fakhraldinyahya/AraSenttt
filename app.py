from flask import Flask, render_template, request, jsonify, send_file
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

from config import Config

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

from utils.manager import model_manager

# Note: Models are now lazy-loaded by model_manager
def load_models():
    return model_manager.load_models()

# التأكد من وجود المجلدات المطلوبة
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

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

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("Received file upload request")
        # التحقق من وجود النماذج فقط عند محاولة الرفع
        if not load_models():
            return jsonify({
                'error': '❌ النماذج غير موجودة! يجب تضمين ملفات النماذج في مجلد models أولاً',
                'models_missing': True
            }), 400
            
        if 'file' not in request.files:
            return jsonify({'error': 'لم يتم رفع أي ملف'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'الملف فارغ'}), 400
        
        if file and allowed_file(file.filename):
            original_filename = file.filename
            file_ext = os.path.splitext(original_filename)[1].lower()
            
            # تأمين اسم الملف مع الحفاظ على الحروف العربية
            filename = secure_filename_arabic(original_filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            print(f"File saved to {filepath}")
            print(f"Uploaded file: {filepath}")
            print(f"Extension detected: {file_ext}")

            if file_ext == ".csv":
                df = pd.read_csv(filepath)

            elif file_ext == ".xlsx":
                df = pd.read_excel(filepath, engine="openpyxl")

            elif file_ext == ".xls":
                df = pd.read_excel(filepath, engine="xlrd")

            else:
                return jsonify({"error": f"Unsupported file format: {file_ext}"}), 400
            
            raw_path = f'data/raw/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{filename}'
            df.to_csv(raw_path, index=False)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'rows': len(df),
                'columns': list(df.columns)
            })
        
        return jsonify({'error': 'نوع الملف غير مسموح'}), 400
    
    return render_template('upload.html')

@app.route('/analyze/<filename>', methods=['POST'])
def analyze_file(filename):
    processed_path = f'data/processed/analyzed_{filename}'
    
    # Use Cache if exists
    if os.path.exists(processed_path):
        print(f"Loading cached results for: {filename}")
        df_cached = pd.read_csv(processed_path)
        results = []
        for _, row in df_cached.iterrows():
            try:
                aspects = ast.literal_eval(row['aspects'])
            except:
                aspects = []
            results.append({
                'original_text': row['original_text'],
                'cleaned_text': row['cleaned_text'],
                'aspects': aspects,
                'overall_sentiment': row['overall_sentiment'],
                'confidence': float(row['confidence'])
            })
        return jsonify({'success': True, 'results': results, 'stats': _calculate_stats(results)})

    # Load file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if filename.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    
    text_column = next((col for col in ['text', 'review', 'comment', 'نص', 'تقييم'] if col in df.columns), None)
    if text_column is None:
        return jsonify({'error': 'لم يتم العثور على عمود النص في الملف'}), 400
    
    # Process Batch
    texts = df[text_column].fillna("").astype(str).tolist()
    results = model_manager.analyze_batch(texts)
    
    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(processed_path, index=False)
    
    return jsonify({
        'success': True,
        'results': results,
        'stats': _calculate_stats(results)
    })

def _calculate_stats(results):
    return {
        'total_reviews': len(results),
        'sentiment_distribution': {
            'إيجابي': sum(1 for r in results if r['overall_sentiment'] == 'إيجابي'),
            'سلبي': sum(1 for r in results if r['overall_sentiment'] == 'سلبي'),
            'محايد': sum(1 for r in results if r['overall_sentiment'] == 'محايد')
        }
    }

@app.route('/results/<filename>')
def get_results(filename):
    """جلب النتائج المحللة مسبقاً للداش بورد"""
    return analyze_file(filename)

@app.route('/dashboard/<filename>')
def dashboard(filename):
    return render_template('dashboard.html', filename=filename)

@app.route('/summary/<filename>')
def summary(filename):
    filepath = f'data/processed/analyzed_{filename}'
    
    if not os.path.exists(filepath):
        return render_template('summary.html', error='الملف غير موجود')
    
    df = pd.read_csv(filepath)
    
    total_reviews = len(df)
    positive_count = len(df[df['overall_sentiment'] == 'إيجابي'])
    negative_count = len(df[df['overall_sentiment'] == 'سلبي'])
    neutral_count = len(df[df['overall_sentiment'] == 'محايد'])
    
    positive_pct = (positive_count / total_reviews) * 100
    negative_pct = (negative_count / total_reviews) * 100
    neutral_pct = (neutral_count / total_reviews) * 100
    
    return render_template('summary.html',
                         filename=filename,
                         total_reviews=total_reviews,
                         positive_count=positive_count,
                         negative_count=negative_count,
                         neutral_count=neutral_count,
                         positive_pct=round(positive_pct, 1),
                         negative_pct=round(negative_pct, 1),
                         neutral_pct=round(neutral_pct, 1))

@app.route('/download/<filename>')
def download_file(filename):
    filepath = f'data/processed/analyzed_{filename}'
    
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