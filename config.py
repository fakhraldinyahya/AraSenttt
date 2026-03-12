import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'arasent_secret_key_2024'
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    DATA_RAW_FOLDER = os.path.join(basedir, 'data', 'raw')
    DATA_PROCESSED_FOLDER = os.path.join(basedir, 'data', 'processed')
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'instance', 'site.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    ASPECT_MODEL_PATH = os.path.join(basedir, 'models', 'aspect_extractor_final')
    SENTIMENT_MODEL_PATH = os.path.join(basedir, 'models', 'sentiment_model_final')
    
    ARABERT_MODEL = 'aubmindlab/bert-base-arabertv2'
    
    ASPECTS = {
        'food': 'الطعام',
        'service': 'الخدمة',
        'price': 'السعر',
        'ambience': 'الأجواء',
        'location': 'الموقع',
        'cleanliness': 'النظافة',
        'delivery': 'التوصيل'
    }
    
    SENTIMENTS = {
        0: 'إيجابي',
        1: 'سلبي',
        2: 'محايد'
    }