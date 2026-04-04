import os
import time
import numpy as np
from typing import Optional, Dict, Tuple
from groq import Groq
from .base_scorer import BaseLLMScorer

class GroqLLMScorer(BaseLLMScorer):
    def __init__(self, model_key: str, config, prompt_strategy: str = 'basic', 
                 api_key: Optional[str] = None):
        super().__init__(model_key, config, prompt_strategy)
        
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API Key is required.")

        self.client = Groq(api_key=api_key)
        print(f"✅ Initialized Groq Scorer: {self.config['description']}")

    def generate_response(self, prompt: str) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert linguist. Return ONLY a number or 'Rating: X.X'."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    model=self.config['name'],
                    temperature=self.config.get('temperature', 0.0),
                    max_tokens=self.config.get('max_tokens', 150),
                )
                return chat_completion.choices[0].message.content.strip()
            
            except Exception as e:
                err_msg = str(e)
                err_lower = err_msg.lower()
                
                # --- IN LỖI CHI TIẾT RA CONSOLE ---
                print(f"\n🔍 [DEBUG API ERROR] ID: {self.model_key} | Lần thử: {attempt+1}")
                print(f"   >>> Chi tiết lỗi: {err_msg}")

                # 1. Lỗi Rate Limit (429) - Tiếp tục thử lại
                if "429" in err_lower or "rate_limit" in err_lower:
                    wait_time = 25 * (attempt + 1)
                    print(f"   ⚠️ Đang đợi {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                # 2. Với các lỗi khác, trả về chuỗi kèm Message lỗi để Evaluator in ra
                return f"ERROR_FATAL: {err_msg}"
        
        return "ERROR_FATAL: Max retries exceeded after Rate Limit"

    def score_plausibility(self, sample: Dict) -> Tuple[float, str]:
        prompt = self.create_prompt(sample)
        res = self.generate_response(prompt)
        
        # Kiểm tra xem có phải chuỗi lỗi không
        if "ERROR_FATAL" in res:
            return -1.0, res
            
        score = self.extract_rating(res)
        return float(score), res