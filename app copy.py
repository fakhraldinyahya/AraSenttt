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

from config import Config
from utils.preprocessing import ArabicTextPreprocessor
from utils.aspect_extractor import AspectExtractor
from utils.sentiment_classifier import SentimentClassifier

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# تهيئة المكونات
preprocessor = ArabicTextPreprocessor()
aspect_extractor = AspectExtractor(model_path='models/aspect_extraction_model')
sentiment_classifier = SentimentClassifier(model_path='models/sentiment_model')

# التأكد من وجود المجلدات المطلوبة
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'لم يتم رفع أي ملف'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'الملف فارغ'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
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
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if filename.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    
    text_column = None
    for col in ['text', 'review', 'comment', 'نص', 'تقييم']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        return jsonify({'error': 'لم يتم العثور على عمود النص في الملف'}), 400
    
    results = []
    for idx, row in df.iterrows():
        text = str(row[text_column])
        cleaned_text = preprocessor.clean_text(text)
        aspects = aspect_extractor.predict_aspects(cleaned_text)
        
        aspect_sentiments = []
        for aspect in aspects:
            sentiment = sentiment_classifier.predict_sentiment(cleaned_text, aspect)
            aspect_sentiments.append({
                'aspect': aspect,
                'sentiment': sentiment['sentiment'],
                'confidence': sentiment['confidence']
            })
        
        overall_sentiment = sentiment_classifier.predict_sentiment(cleaned_text)
        
        results.append({
            'original_text': text,
            'cleaned_text': cleaned_text,
            'aspects': aspect_sentiments,
            'overall_sentiment': overall_sentiment['sentiment'],
            'confidence': overall_sentiment['confidence']
        })
    
    results_df = pd.DataFrame(results)
    processed_path = f'data/processed/analyzed_{filename}'
    results_df.to_csv(processed_path, index=False)
    
    return jsonify({
        'success': True,
        'results': results,
        'stats': {
            'total_reviews': len(results),
            'sentiment_distribution': {
                'إيجابي': sum(1 for r in results if r['overall_sentiment'] == 'إيجابي'),
                'سلبي': sum(1 for r in results if r['overall_sentiment'] == 'سلبي'),
                'محايد': sum(1 for r in results if r['overall_sentiment'] == 'محايد')
            }
        }
    })

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)