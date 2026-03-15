"""
utils/multitask_analyzer.py
============================
محلل AraSent متعدد المهام (Multi-Task)
يعتمد على النموذج المدرب: arasant_multitask_model_last_v2
التدريب مبني على AraBERT v2 مع مهمتين:
  1. كشف وجود الفئة (Aspect Existence) - 7 فئات
  2. تصنيف المشاعر للفئات الموجودة (Sentiment) - 3 تصنيفات لكل فئة
"""

import os
import re
import ast
import json
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig

# ============================================================
# ثوابت
# ============================================================
CATEGORIES = ['FOOD', 'SERVICE', 'PRICE', 'AMBIENCE', 'LOCATION', 'CLEANLINESS', 'DELIVERY']

# ترجمة الفئات للعربية (لتوافق الداشبورد)
CATEGORY_AR = {
    'FOOD':        'الطعام',
    'SERVICE':     'الخدمة',
    'PRICE':       'السعر',
    'AMBIENCE':    'الأجواء',
    'LOCATION':    'الموقع',
    'CLEANLINESS': 'النظافة',
    'DELIVERY':    'التوصيل',
}

# ترجمة المشاعر للعربية (لتوافق الداشبورد)
SENTIMENT_AR = {
    'positive': 'إيجابي',
    'negative': 'سلبي',
    'neutral':  'محايد',
}

# الخريطة العكسية: رقم → نص إنجليزي
SENTIMENT_IDX = {0: 'negative', 1: 'neutral', 2: 'positive'}

# عتبة الثقة لوجود الفئة - 0.45 لتجنب التشخيصات الوهمية (false positives < 0.50)
DEFAULT_THRESHOLD = 0.45


# ============================================================
# بنية النموذج (نفس كود التدريب)
# ============================================================
class AraSentMultiTaskModel(nn.Module):
    """
    نموذج متعدد المهام:
      - المهمة 1: كشف وجود الفئة (7 binary outputs)
      - المهمة 2: تصنيف المشاعر (7 * 3 outputs)
    """
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_config(config)
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.existence_classifier = nn.Linear(self.hidden_size, 7)
        self.sentiment_classifier = nn.Linear(self.hidden_size, 7 * 3)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = self.dropout(outputs.last_hidden_state[:, 0, :])
        return {
            'existence_logits': self.existence_classifier(pooled),
            'sentiment_logits': self.sentiment_classifier(pooled),
        }


# ============================================================
# تنظيف النص (نفس منطق التدريب)
# ============================================================
def _clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = str(text).replace('\n', ' ').replace('\r', ' ').strip()
    if not text:
        return ""
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)   # إزالة التشكيل
    text = re.sub(r'[إأآا]', 'ا', text)                         # توحيد الألف
    text = re.sub(r'(.)\1+', r'\1\1', text)                    # تقليل التكرار
    return text.strip()


# ============================================================
# الكلاس الرئيسي
# ============================================================
class MultiTaskAnalyzer:
    """
    واجهة التحليل متعددة المهام.
    يحمل الموديل مرة واحدة ويحتفظ به في الذاكرة (singleton-friendly).
    """

    def __init__(self, model_dir: str = 'models/arasant_multitask_model_last_v2',
                 threshold: float = DEFAULT_THRESHOLD):
        self.model_dir = model_dir
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is not None:
            return True
        try:
            print(f"[MultiTask] Loading model from {self.model_dir} ...")

            # قراءة model_info.json لمعرفة اسم base model
            info_path = os.path.join(self.model_dir, 'model_info.json')
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            # base_model_name = info.get('base_model', 'aubmindlab/bert-base-arabertv2')

            # السر الذي اكتشفناه في test.py: من أجل عدم تحميل النموذج الأساسي من الإنترنت نهائياً،
            # نستخدم from_config لإنشاء معمارية فارغة، ثم لاحقاً يتم تحميل الأوزان المحلية فوقها.
            # تحميل الكونفيج من المجلد
            self.config = AutoConfig.from_pretrained(self.model_dir)

            # تحميل التوكنيزر من المجلد
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

            # بناء بنية النموذج
            self.model = AraSentMultiTaskModel(self.config) # Changed to pass self.config

            # تحميل الأوزان
            weights_path = os.path.join(self.model_dir, 'model_weights.pth')
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            print(f"[MultiTask] Model loaded successfully on {self.device} ")
            return True
        except Exception as e:
            print(f"[MultiTask] Error loading model: {e}")
            self.model = None
            self.tokenizer = None
            return False

    def _predict_single(self, text: str) -> dict:
        """تحليل نص واحد وإرجاع نتائج خام."""
        cleaned = _clean_text(text)
        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding='max_length'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs)
            existence_logits = out['existence_logits'][0].cpu().numpy()
            sentiment_logits = out['sentiment_logits'][0].cpu().numpy()

        # تحويل logits إلى احتماليات
        existence_probs = 1.0 / (1.0 + np.exp(-existence_logits))

        aspects = []
        for i, cat in enumerate(CATEGORIES):
            if existence_probs[i] > self.threshold:
                sent_scores = sentiment_logits[i * 3: (i + 1) * 3]
                sent_idx = int(np.argmax(sent_scores))
                sent_en = SENTIMENT_IDX[sent_idx]
                aspects.append({
                    'aspect':     CATEGORY_AR[cat],        # اسم عربي للداشبورد
                    'sentiment':  SENTIMENT_AR[sent_en],   # مشاعر عربية للداشبورد
                    'confidence': float(existence_probs[i]),
                    'implicit':   False,
                    'evidence':   None,
                })

        # ترتيب حسب الثقة
        aspects.sort(key=lambda x: x['confidence'], reverse=True)
        return {'cleaned': cleaned, 'aspects': aspects}

    def _compute_overall_sentiment(self, aspects: list) -> tuple:
        """
        حساب المشاعر الكلية من الجوانب الموجودة.
        يُرجع (sentiment_ar, confidence).
        """
        if not aspects:
            return 'محايد', 0.5

        counts = {'إيجابي': 0, 'سلبي': 0, 'محايد': 0}
        total_conf = 0.0
        for a in aspects:
            counts[a['sentiment']] = counts.get(a['sentiment'], 0) + 1
            total_conf += a['confidence']

        avg_conf = total_conf / len(aspects)
        overall = max(counts, key=counts.get)
        return overall, round(avg_conf, 4)

    def analyze_batch(self, texts: list) -> list:
        """
        تحليل مجموعة نصوص وإرجاع قائمة نتائج بنفس الصيغة
        التي يتوقعها الداشبورد والكود القائم.
        """
        if self.model is None:
            if not self.load():
                raise RuntimeError("لم يتم تحميل موديل MultiTask. تأكد من وجود ملفات الموديل.")

        results = []
        for text in texts:
            try:
                raw = self._predict_single(str(text))
                overall_sent, conf = self._compute_overall_sentiment(raw['aspects'])
                results.append({
                    'original_text':     text,
                    'cleaned_text':      raw['cleaned'],
                    'aspects':           raw['aspects'],
                    'overall_sentiment': overall_sent,
                    'confidence':        conf,
                    'provider':          'local_multitask',
                    'logic_explanation': 'تحليل متعدد المهام عبر AraBERT - 7 فئات',
                })
            except Exception as e:
                print(f"[MultiTask] Error analyzing text: {e}")
                results.append({
                    'original_text':     text,
                    'cleaned_text':      _clean_text(str(text)),
                    'aspects':           [],
                    'overall_sentiment': 'محايد',
                    'confidence':        0.0,
                    'provider':          'local_multitask',
                    'logic_explanation': f'خطأ في التحليل: {str(e)}',
                })
        return results
