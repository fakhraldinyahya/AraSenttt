import hashlib
import json
import os
import re

import pandas as pd


POSITIVE_NAME_KEYWORDS = {
    'text': 10,
    'comment': 12,
    'comments': 12,
    'review': 12,
    'reviews': 12,
    'feedback': 11,
    'response': 8,
    'responses': 8,
    'message': 7,
    'notes': 7,
    'note': 7,
    'description': 6,
    'content': 6,
    'body': 5,
    'opinion': 10,
    'complaint': 8,
    'suggestion': 8,
    'suggestions': 8,
    'answer': 7,
    'answers': 7,
    'commentaire': 12,
    'تعليق': 12,
    'التعليق': 12,
    'تعليقات': 12,
    'التعليقات': 12,
    'ملاحظات': 11,
    'ملاحظه': 11,
    'ملاحظة': 11,
    'راي': 10,
    'الرأي': 10,
    'الراي': 10,
    'اراء': 10,
    'آراء': 10,
    'تقييم': 10,
    'التقييم': 10,
    'تجربه': 8,
    'تجربة': 8,
    'اقتراح': 8,
    'اقتراحات': 8,
    'شكوي': 8,
    'شكوى': 8,
    'نص': 8,
    'النص': 8,
    'الوصف': 6,
    'محتوي': 6,
    'محتوى': 6,
    'اجابه': 7,
    'إجابة': 7,
    'استجابه': 7,
    'استجابة': 7,
}

NEGATIVE_NAME_KEYWORDS = {
    'id': -10,
    'uuid': -10,
    'date': -8,
    'time': -8,
    'timestamp': -8,
    'email': -8,
    'phone': -8,
    'mobile': -8,
    'rating': -7,
    'score': -7,
    'stars': -7,
    'rank': -6,
    'count': -6,
    'amount': -6,
    'price': -6,
    'branch': -5,
    'city': -5,
    'country': -5,
    'gender': -5,
    'age': -5,
    'رقم': -10,
    'التاريخ': -8,
    'تاريخ': -8,
    'وقت': -8,
    'البريد': -8,
    'ايميل': -8,
    'جوال': -8,
    'هاتف': -8,
    'تقييم رقمي': -8,
    'عدد': -6,
    'سعر': -6,
    'فرع': -5,
    'مدينه': -5,
    'مدينة': -5,
    'عمر': -5,
    'جنس': -5,
}


def read_uploaded_dataframe(filepath):
    file_ext = os.path.splitext(filepath)[1].lower()
    if file_ext == ".csv":
        return pd.read_csv(filepath)
    if file_ext == ".xlsx":
        return pd.read_excel(filepath, engine="openpyxl")
    if file_ext == ".xls":
        return pd.read_excel(filepath, engine="xlrd")
    raise ValueError(f"Unsupported file format: {file_ext}")


def normalize_column_label(value):
    normalized = str(value or "").strip().lower()
    normalized = normalized.translate(str.maketrans({
        'أ': 'ا',
        'إ': 'ا',
        'آ': 'ا',
        'ى': 'ي',
        'ؤ': 'و',
        'ئ': 'ي',
        'ة': 'ه',
    }))
    normalized = re.sub(r'[_\-/]+', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.strip()


def build_text_column_candidates(df):
    candidates = []
    total_rows = max(len(df.index), 1)

    for column in df.columns:
        series = df[column]
        normalized_name = normalize_column_label(column)
        score = 0
        reasons = []

        for keyword, weight in POSITIVE_NAME_KEYWORDS.items():
            normalized_keyword = normalize_column_label(keyword)
            if normalized_keyword and normalized_keyword in normalized_name:
                score += weight
                reasons.append(f"اسم العمود يوحي بأنه نصي ({keyword})")

        for keyword, weight in NEGATIVE_NAME_KEYWORDS.items():
            normalized_keyword = normalize_column_label(keyword)
            if normalized_keyword and normalized_keyword in normalized_name:
                score += weight
                reasons.append(f"اسم العمود أقرب لبيانات وصفية ({keyword})")

        non_null = series.dropna()
        non_empty = non_null.astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
        non_empty = non_empty[non_empty != ""]

        non_empty_count = int(non_empty.shape[0])
        if not non_empty_count:
            candidates.append({
                'name': column,
                'score': score - 20,
                'reason': 'العمود فارغ',
                'non_empty_count': 0,
                'fill_ratio': 0,
            })
            continue

        sample = non_empty.head(40)
        lengths = sample.str.len()
        word_counts = sample.str.split().str.len()
        avg_length = float(lengths.mean()) if not lengths.empty else 0.0
        avg_words = float(word_counts.mean()) if not word_counts.empty else 0.0
        fill_ratio = non_empty_count / total_rows
        unique_ratio = float(sample.nunique() / max(len(sample), 1))
        digit_only_ratio = float(sample.str.fullmatch(r'[\d\W_]+', na=False).mean())

        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            score += 4
            reasons.append('نوع العمود نصي')
        elif pd.api.types.is_numeric_dtype(series):
            score -= 12
            reasons.append('نوع العمود رقمي')

        if avg_length >= 60:
            score += 10
            reasons.append('متوسط النص طويل')
        elif avg_length >= 25:
            score += 7
            reasons.append('متوسط النص مناسب للتعليقات')
        elif avg_length >= 10:
            score += 3

        if avg_words >= 8:
            score += 8
            reasons.append('القيم تحتوي على جمل متعددة الكلمات')
        elif avg_words >= 4:
            score += 5
        elif avg_words >= 2:
            score += 2

        if fill_ratio >= 0.5:
            score += 3
        elif fill_ratio < 0.15:
            score -= 4

        if unique_ratio >= 0.7:
            score += 2

        if digit_only_ratio >= 0.6:
            score -= 10
            reasons.append('معظم القيم ليست نصوصًا حرة')

        candidates.append({
            'name': column,
            'score': round(score, 2),
            'reason': '، '.join(dict.fromkeys(reasons)) if reasons else 'تقييم محتوى العمود',
            'non_empty_count': non_empty_count,
            'fill_ratio': round(fill_ratio, 2),
        })

    candidates.sort(key=lambda item: item['score'], reverse=True)
    return candidates


def detect_text_column(df):
    candidates = build_text_column_candidates(df)
    if not candidates:
        return {'selected': None, 'candidates': [], 'requires_manual_selection': True}

    top_candidate = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    score_gap = top_candidate['score'] - runner_up['score'] if runner_up else top_candidate['score']
    selected = top_candidate['name'] if top_candidate['score'] >= 6 and score_gap >= 2 else None

    return {
        'selected': selected,
        'candidates': candidates[:5],
        'requires_manual_selection': selected is None,
    }


def build_analysis_cache_key(texts, model_name):
    payload = {
        'version': 1,
        'model': model_name,
        'texts': texts,
    }
    canonical_payload = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(canonical_payload.encode('utf-8')).hexdigest()
