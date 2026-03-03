import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class AspectExtractor:
    def __init__(self, model_path='models/aspect_extractor_final'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def predict_aspects(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        aspects = []
        current_aspect = []
        
        for token, pred in zip(tokens, predictions):
            if pred == 1:
                if current_aspect:
                    aspects.append(self.tokenizer.convert_tokens_to_string(current_aspect))
                current_aspect = [token]
            elif pred == 2 and current_aspect:
                current_aspect.append(token)
            else:
                if current_aspect:
                    aspects.append(self.tokenizer.convert_tokens_to_string(current_aspect))
                    current_aspect = []
        
        if current_aspect:
            aspects.append(self.tokenizer.convert_tokens_to_string(current_aspect))
        
        return aspects

    def predict_aspects_batch(self, texts: list):
        if not texts:
            return []
            
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2).cpu().numpy()
        
        all_aspects = []
        for i, text in enumerate(texts):
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][i])
            preds = predictions[i]
            
            aspects = []
            current_aspect = []
            
            for token, pred in zip(tokens, preds):
                if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                    continue
                
                if pred == 1: # B-ASPECT
                    if current_aspect:
                        aspects.append(self.tokenizer.convert_tokens_to_string(current_aspect))
                    current_aspect = [token]
                elif pred == 2 and current_aspect: # I-ASPECT
                    current_aspect.append(token)
                else:
                    if current_aspect:
                        aspects.append(self.tokenizer.convert_tokens_to_string(current_aspect))
                        current_aspect = []
            
            if current_aspect:
                aspects.append(self.tokenizer.convert_tokens_to_string(current_aspect))
            all_aspects.append(aspects)
            
        return all_aspects