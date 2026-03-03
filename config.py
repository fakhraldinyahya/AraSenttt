import os

class Config:
    SECRET_KEY = 'arasent_secret_key_2024'
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    
    ASPECT_MODEL_PATH = 'models/aspect_extractor_final'
    SENTIMENT_MODEL_PATH = 'models/sentiment_model_final'
    
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