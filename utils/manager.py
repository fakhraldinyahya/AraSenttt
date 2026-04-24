import threading

from utils.arasent_analyzer import AraSentAnalyzer


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
        self.arasant_analyzer = None
        self.loading = False
        self._initialized = True

    def load_arasant_model(self, force=False):
        if self.arasant_analyzer is not None and not force:
            return True

        try:
            self.loading = True
            self.arasant_analyzer = AraSentAnalyzer(
                model_dir='models/arasant_multitask_model_last_v2'
            )
            return self.arasant_analyzer.load()
        except Exception as e:
            print(f"Error Loading AraSent Model: {e}")
            self.arasant_analyzer = None
            return False
        finally:
            self.loading = False

    def analyze_batch_arasant(self, texts):
        if self.arasant_analyzer is None:
            if not self.load_arasant_model():
                raise RuntimeError("فشل تحميل موديل AraSent")
        return self.arasant_analyzer.analyze_batch(texts)


model_manager = ModelManager()
