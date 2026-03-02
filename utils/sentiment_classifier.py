import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentClassifier:
    def __init__(self, model_path='models/sentiment_model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.sentiment_labels = {
            0: 'إيجابي',
            1: 'سلبي', 
            2: 'محايد'
        }
        
    def predict_sentiment(self, text: str, aspect: str = None):
        if aspect:
            input_text = f"{aspect} [SEP] {text}"
        else:
            input_text = text
        
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
            confidence = probabilities[prediction].item()
        
        return {
            'sentiment': self.sentiment_labels[prediction],
            'confidence': confidence
        }