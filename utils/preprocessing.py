import re
import pandas as pd
import arabic_reshaper
from bidi.algorithm import get_display

class ArabicTextPreprocessor:
    def __init__(self):
        self.diacritics = re.compile(r'[ًٌٍَُِّْ]')
        self.punctuations = re.compile(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]')
        self.repeated_chars = re.compile(r'(.)\1{2,}')
        
    def remove_diacritics(self, text):
        return self.diacritics.sub('', str(text))
    
    def normalize_arabic(self, text):
        text = re.sub('[إأآا]', 'ا', text)
        text = re.sub('ة', 'ه', text)
        text = re.sub('ى', 'ي', text)
        text = re.sub('ؤ', 'ء', text)
        text = re.sub('ئ', 'ء', text)
        return text
    
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        
        text = str(text)
        text = self.remove_diacritics(text)
        text = self.normalize_arabic(text)
        text = self.punctuations.sub(' ', text)
        text = ' '.join(text.split())
        
        return text.strip()