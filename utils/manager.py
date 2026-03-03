import os
import threading
from utils.preprocessing import ArabicTextPreprocessor
from utils.aspect_extractor import AspectExtractor
from utils.sentiment_classifier import SentimentClassifier

class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.preprocessor = None
        self.aspect_extractor = None
        self.sentiment_classifier = None
        self.loading = False
        self._initialized = True

    def load_models(self, force=False):
        if (self.preprocessor and self.aspect_extractor and self.sentiment_classifier) and not force:
            return True

        self.loading = True
        try:
            print("🚀 Loading AraSent Models...")
            self.preprocessor = ArabicTextPreprocessor()
            
            # Use paths from config or defaults
            aspect_path = 'models/aspect_extractor_final'
            sentiment_path = 'models/sentiment_model_final'
            
            self.aspect_extractor = AspectExtractor(model_path=aspect_path)
            self.sentiment_classifier = SentimentClassifier(model_path=sentiment_path)
            
            print("✅ Models Loaded Successfully!")
            return True
        except Exception as e:
            print(f"❌ Error Loading Models: {e}")
            return False
        finally:
            self.loading = False

    def analyze_batch(self, texts):
        """
        Analyze a batch of texts for both aspects and sentiment.
        Optimized to reduce redundant processing.
        """
        if not self.preprocessor:
            self.load_models()

        results = []
        cleaned_texts = [self.preprocessor.clean_text(str(t)) for t in texts]
        
        # 1. Batch predict aspects (Now fully batch!)
        all_aspects_list = self.aspect_extractor.predict_aspects_batch(cleaned_texts)
        
        for i, text in enumerate(texts):
            cleaned = cleaned_texts[i]
            aspects = all_aspects_list[i]
            
            # Pre-collect data for overall sentiment batch
            results.append({
                'original_text': text,
                'cleaned_text': cleaned,
                'aspects_raw': aspects,
                'aspect_sentiments': []
            })

        # 2. Batch predict overall sentiment
        overall_sentiments = self.sentiment_classifier.predict_sentiment_batch(cleaned_texts)
        
        # 3. Handle Aspect-based Sentiment in batches if possible
        # Collect all (text, aspect) pairs for a single large batch call
        all_aspect_texts = []
        all_aspect_labels = []
        aspect_map = [] # To map results back

        for i, res in enumerate(results):
            for aspect in res['aspects_raw']:
                all_aspect_texts.append(res['cleaned_text'])
                all_aspect_labels.append(aspect)
                aspect_map.append((i, aspect))

        if all_aspect_texts:
            batch_aspect_sentiments = self.sentiment_classifier.predict_sentiment_batch(all_aspect_texts, all_aspect_labels)
            for (res_idx, aspect_name), sentiment_res in zip(aspect_map, batch_aspect_sentiments):
                results[res_idx]['aspect_sentiments'].append({
                    'aspect': aspect_name,
                    'sentiment': sentiment_res['sentiment'],
                    'confidence': sentiment_res['confidence']
                })

        # Final Formatting
        final_results = []
        for i, res in enumerate(results):
            final_results.append({
                'original_text': res['original_text'],
                'cleaned_text': res['cleaned_text'],
                'aspects': res['aspect_sentiments'],
                'overall_sentiment': overall_sentiments[i]['sentiment'],
                'confidence': overall_sentiments[i]['confidence']
            })
            
        return final_results

# Singleton instance
model_manager = ModelManager()
