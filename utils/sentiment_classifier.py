import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentClassifier:
    def __init__(self, model_path='models/sentiment_model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # نقرأ الترتيب الحقيقي من الموديل نفسه
        self.id2label = self.model.config.id2label
        
        # خريطة تحويل للعربية
        self.arabic_map = {
            "positive": "إيجابي",
            "negative": "سلبي",
            "neutral": "محايد"
        }

    def predict_sentiment(self, text: str, aspect: str = None):
        
        # إذا فيه جانب نستخدم صيغة aspect [SEP] text
        if aspect:
            input_text = f"{aspect} [SEP] {text}"
        else:
            input_text = text
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
            confidence = probabilities[prediction].item()
        
        # نأخذ اسم اللابل الحقيقي من الموديل
        label_name = self.model.config.id2label.get(
        str(prediction),
        self.model.config.id2label.get(prediction)
)
        
        # نحوله للعربية
        arabic_label = self.arabic_map.get(label_name, label_name)
        
        return {
            'sentiment': arabic_label,
            'confidence': confidence
        }