from google import genai
from .base_scorer import BaseLLMScorer
import logging

class GeminiLLMScorer(BaseLLMScorer):
    def __init__(self, model_key, config, api_key, prompt_strategy="basic"):
        # 1. Khởi tạo lớp cha (BaseLLMScorer thường chỉ giữ logic prompt/rating)
        super().__init__(model_key, prompt_strategy) 
        
        # 2. Lưu config vào một thuộc tính riêng để tránh xung đột với Base class
        self.model_config = config 
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = config.get("name", model_key)
        self.temperature = config.get("temperature", 0.1)

    def generate(self, prompt):
        try:
            # Sửa lỗi: Dùng self.model_config thay vì self.config
            generate_config = {
                'temperature': self.temperature,
                'max_output_tokens': self.model_config.get('max_tokens', 100)
            }
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generate_config
            )
            return response.text if response.text else ""
        except Exception as e:
            # Nếu bị lỗi, in ra toàn bộ lỗi để debug
            logging.error(f"Lỗi API Gemini: {e}")
            return f"Error: {str(e)}"

    def score_plausibility(self, sample):
        prompt = self.create_prompt(sample)
        raw_response = self.generate(prompt)
        rating = self.extract_rating(raw_response)
        return rating, raw_response