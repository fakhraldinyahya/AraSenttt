from google import genai
import json
import re
import time

class GeminiAnalyzer:
    def __init__(self, api_key, model_name="gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.aspect_map = {
            "Food": "الطعام",
            "Service": "الخدمة",
            "Price": "السعر",
            "Ambience": "الأجواء",
            "Location": "الموقع",
            "Cleanliness": "النظافة",
            "Delivery": "التوصيل"
        }
        self.sentiment_map = {
            "Positive": "إيجابي",
            "Negative": "سلبي",
            "Neutral": "محايد",
            "Not_Mentioned": None
        }

    def analyze_batch(self, texts):
        if not texts:
            return []
            
        # Optimization: Process in batches of 10 to 15 for Gemini efficiency
        batch_size = 10
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self._process_single_batch(batch)
            all_results.extend(batch_results)
            time.sleep(1) # Rate limit safety
            
        return all_results

    def _process_single_batch(self, batch):
        system_instruction = f"""
        You are an expert Arabic restaurant review sentiment analyst specializing in:
        1. Aspect-Based Sentiment Analysis (ABSA): Identifying sentiment for specific aspects (Food, Service, etc.).
        2. Implicit Sentiment Detection: Detecting sentiment from indirect language, sarcasm, or context.
        3. Overall Sentiment: The general tone of the review.

        Analyze the provided batch of {len(batch)} reviews and return a JSON list of objects.
        For each review:
        - Identify aspects mentioned (even if implicit).
        - Determine overall sentiment and confidence.
        - For EVERY identified aspect, provide:
            1. "sentiment": Positive|Negative|Neutral|Not_Mentioned
            2. "confidence": A float between 0 and 1.
            3. "implicit": true if the sentiment is implied/indirect, false if explicit.
            4. "evidence": A VERY CONCISE Arabic snippet from the text (max 50 chars) that supports this (الشهادة).
        - Provide a 'logic_summary' in Arabic explaining the analysis.

        JSON structure:
        {{
          "aspects": {{
            "Food": {{"sentiment": "...", "confidence": 0.0, "implicit": false, "evidence": "..."}},
            "Service": {{"sentiment": "...", "confidence": 0.0, "implicit": false, "evidence": "..."}},
            "Price": {{"sentiment": "...", "confidence": 0.0, "implicit": false, "evidence": "..."}},
            "Ambience": {{"sentiment": "...", "confidence": 0.0, "implicit": false, "evidence": "..."}},
            "Location": {{"sentiment": "...", "confidence": 0.0, "implicit": false, "evidence": "..."}},
            "Cleanliness": {{"sentiment": "...", "confidence": 0.0, "implicit": false, "evidence": "..."}},
            "Delivery": {{"sentiment": "...", "confidence": 0.0, "implicit": false, "evidence": "..."}}
          }},
          "overall_sentiment": "Positive|Negative|Neutral",
          "overall_confidence": 0.0,
          "logic_summary": "شرح بالعربية لسبب اختيار هذا المشعر"
        }}
        Return EXACTLY {len(batch)} objects in a list.
        """
        
        prompt = f"Analyze these reviews: {json.dumps(batch, ensure_ascii=False)}"
        
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "system_instruction": system_instruction,
                        "response_mime_type": "application/json",
                        "temperature": 0.1
                    }
                )
                
                text = response.text
                if not text:
                    raise Exception("Empty response from Gemini")
                
                # Cleanup common JSON issues if any
                text = text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                
                # Use regex to find the actual JSON list if there's surrounding text
                start = text.find("[")
                end = text.rfind("]") + 1
                if start != -1 and end != -1:
                    text = text[start:end]

                raw_results = json.loads(text)
                if not isinstance(raw_results, list):
                    raw_results = [raw_results] # Fallback for single object
                
                processed_results = []
                for idx, data in enumerate(raw_results):
                    # Ensure we don't go out of bounds if Gemini returns fewer/more
                    if idx >= len(batch): break
                    
                    text = batch[idx]
                    processed_aspects = []
                    for eng_aspect, ar_aspect in self.aspect_map.items():
                        aspect_data = data.get("aspects", {}).get(eng_aspect, {})
                        sentiment = aspect_data.get("sentiment")
                        ar_sentiment = self.sentiment_map.get(sentiment)
                        if ar_sentiment:
                            processed_aspects.append({
                                'aspect': ar_aspect,
                                'sentiment': ar_sentiment,
                                'confidence': float(aspect_data.get('confidence', 0.9)),
                                'implicit': aspect_data.get('implicit', False),
                                'evidence': aspect_data.get('evidence', '')
                            })
                    
                    overall_sentiment = self.sentiment_map.get(data.get("overall_sentiment"), "محايد")
                    
                    processed_results.append({
                        'original_text': text,
                        'cleaned_text': text,
                        'aspects': processed_aspects,
                        'overall_sentiment': overall_sentiment,
                        'confidence': float(data.get('overall_confidence', data.get('confidence', 0.9))),
                        'logic_explanation': data.get('logic_summary', ''),
                        'provider': 'gemini'
                    })
                
                # If Gemini returned fewer results, pad with errors
                while len(processed_results) < len(batch):
                    processed_results.append(self._error_result(batch[len(processed_results)]))
                    
                return processed_results
                
            except Exception as e:
                err_msg = str(e).lower()
                # If it's a fatal error like quota or auth, stop retrying and bubble up
                if "quota" in err_msg or "api_key" in err_msg or "invalid" in err_msg:
                    raise e
                
                print(f"Batch Error (Attempt {attempt+1}): {e}")
                if attempt == 2:
                    return [self._error_result(t) for t in batch]
                time.sleep(2)
        
        return [self._error_result(t) for t in batch]

    def _error_result(self, text):
        return {
            'original_text': text,
            'cleaned_text': text,
            'aspects': [],
            'overall_sentiment': 'محايد',
            'confidence': 0.0,
            'provider': 'error'
        }

    def generate_executive_summary(self, results):
        """توليد ملخص تنفيذي احترافي بناءً على نتائج التحليل"""
        if not results:
            return "لا توجد بيانات متاحة لتوليد الملخص."

        # Calculate stats for the prompt
        stats = {a: {"إيجابي": 0, "سلبي": 0, "محايد": 0} for a in self.aspect_map.values()}
        for item in results:
            aspects = item.get("aspects", [])
            for a_data in aspects:
                a_name = a_data.get('aspect')
                s_name = a_data.get('sentiment')
                if a_name in stats and s_name in stats[a_name]:
                    stats[a_name][s_name] += 1

        prompt = f"""
        إليك إحصائيات مراجعات المطعم بناءً على تحليل المشاعر لكل جانب (Aspect-Based Sentiment Analysis):
        {json.dumps(stats, ensure_ascii=False)}

        اكتب ملخصاً تنفيذياً احترافياً ومقنعاً بالعربية (Markdown format) لمدير المطعم يشمل:
        1. نقاط القوة: الجوانب التي تميز المطعم (الأكثر إيجابية).
        2. نقاط الضعف: الجوانب التي تحتاج إلى تحسين فوري (الأكثر سلبية).
        3. اقتراحات التحسين: خطوات عملية بناءً على النتائج.
        
        اجعل الأسلوب راقياً واحترافياً، واستخدم الرموز التعبيرية (Emojis) المناسبة.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Summary Generation Error: {e}")
            return "عذراً، تعذر توليد الملخص الفني حالياً. يرجى مراجعة إحصائيات الأداء أعلاه."
