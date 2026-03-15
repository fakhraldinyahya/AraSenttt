import os
import threading
import time
from utils.preprocessing import ArabicTextPreprocessor
from utils.aspect_extractor import AspectExtractor
from utils.sentiment_classifier import SentimentClassifier
from utils.multitask_analyzer import MultiTaskAnalyzer

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
        # --- الموديلات القديمة ---
        self.preprocessor = None
        self.aspect_extractor = None
        self.sentiment_classifier = None
        # --- الموديل الجديد متعدد المهام ---
        self.multitask_analyzer = None
        self.loading = False
        self._initialized = True

    # ------------------------------------------------------------------
    # تحميل الموديلات القديمة
    # ------------------------------------------------------------------
    def load_models(self, force=False):
        if (self.preprocessor and self.aspect_extractor and self.sentiment_classifier) and not force:
            return True

        self.loading = True
        try:
            print("Loading AraSent Classic Models...")
            self.preprocessor = ArabicTextPreprocessor()

            aspect_path = 'models/aspect_extractor_final'
            sentiment_path = 'models/sentiment_model_final'

            self.aspect_extractor = AspectExtractor(model_path=aspect_path)
            self.sentiment_classifier = SentimentClassifier(model_path=sentiment_path)

            print("Classic Models Loaded Successfully!")
            return True
        except Exception as e:
            print(f"Error Loading Classic Models: {e}")
            return False
        finally:
            self.loading = False

    # ------------------------------------------------------------------
    # تحميل الموديل الجديد متعدد المهام
    # ------------------------------------------------------------------
    def load_multitask(self, force=False):
        if self.multitask_analyzer is not None and not force:
            return True
        try:
            self.multitask_analyzer = MultiTaskAnalyzer(
                model_dir='models/arasant_multitask_model_last_v2'
            )
            return self.multitask_analyzer.load()
        except Exception as e:
            print(f"Error Loading MultiTask Model: {e}")
            self.multitask_analyzer = None
            return False

    # ------------------------------------------------------------------
    # تحليل باستخدام الموديلات القديمة
    # ------------------------------------------------------------------
    def analyze_batch(self, texts):
        """
        تحليل batch بالموديلات التقليدية (AspectExtractor + SentimentClassifier).
        """
        if not self.preprocessor:
            self.load_models()

        total_start = time.time()
        results = []

        print(f"[Classic] Starting analysis for {len(texts)} reviews...")

        # 0. تنظيف النصوص
        clean_start = time.time()
        cleaned_texts = [self.preprocessor.clean_text(str(t)) for t in texts]
        print(f"[Classic] Cleaning took: {time.time() - clean_start:.2f}s")

        # 1. استخراج الجوانب
        aspect_start = time.time()
        all_aspects_list = self.aspect_extractor.predict_aspects_batch(cleaned_texts)
        print(f"[Classic] Aspect Extraction took: {time.time() - aspect_start:.2f}s")

        for i, text in enumerate(texts):
            cleaned = cleaned_texts[i]
            aspects = all_aspects_list[i]
            results.append({
                'original_text': text,
                'cleaned_text': cleaned,
                'aspects_raw': aspects,
                'aspect_sentiments': []
            })

        # 2. المشاعر الكلية
        sentiment_start = time.time()
        overall_sentiments = self.sentiment_classifier.predict_sentiment_batch(cleaned_texts)
        print(f"[Classic] Overall Sentiment took: {time.time() - sentiment_start:.2f}s")

        # 3. مشاعر الجوانب
        all_aspect_texts = []
        all_aspect_labels = []
        aspect_map = []

        for i, res in enumerate(results):
            for aspect in res['aspects_raw']:
                all_aspect_texts.append(res['cleaned_text'])
                all_aspect_labels.append(aspect)
                aspect_map.append((i, aspect))

        if all_aspect_texts:
            aspect_sentiment_start = time.time()
            batch_aspect_sentiments = []

            chunk_size = 16
            for i in range(0, len(all_aspect_texts), chunk_size):
                chunk_texts = all_aspect_texts[i:i + chunk_size]
                chunk_labels = all_aspect_labels[i:i + chunk_size]
                chunk_results = self.sentiment_classifier.predict_sentiment_batch(chunk_texts, chunk_labels)
                batch_aspect_sentiments.extend(chunk_results)

            print(f"[Classic] Aspect Sentiment ({len(all_aspect_texts)} items) took: {time.time() - aspect_sentiment_start:.2f}s")

            for (res_idx, aspect_name), sentiment_res in zip(aspect_map, batch_aspect_sentiments):
                results[res_idx]['aspect_sentiments'].append({
                    'aspect': aspect_name,
                    'sentiment': sentiment_res['sentiment'],
                    'confidence': sentiment_res['confidence']
                })

        print(f"[Classic] Total Time: {time.time() - total_start:.2f}s")

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

    # ------------------------------------------------------------------
    # تحليل باستخدام الموديل الجديد
    # ------------------------------------------------------------------
    def analyze_batch_multitask(self, texts):
        """
        تحليل batch بالموديل الجديد متعدد المهام.
        """
        if self.multitask_analyzer is None:
            if not self.load_multitask():
                raise RuntimeError("فشل تحميل موديل MultiTask")
        return self.multitask_analyzer.analyze_batch(texts)


# Singleton instance
model_manager = ModelManager()
