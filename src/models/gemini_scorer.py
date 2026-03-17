from google import genai
from .base_scorer import BaseLLMScorer
import logging

class GeminiLLMScorer(BaseLLMScorer):
    def __init__(self, model_key, config, api_key, prompt_strategy="basic"):
        # 1. Khởi tạo lớp cha với đầy đủ 3 tham số bắt buộc
        # Việc truyền config vào đây giúp self.create_prompt() hoạt động chính xác
        super().__init__(model_key, config, prompt_strategy) 
        
        # 2. Lưu config vào một thuộc tính riêng để sử dụng nội bộ trong class này
        self.model_config = config 
        
        # 3. Khởi tạo Gemini Client từ SDK mới nhất (google-genai)
        self.client = genai.Client(api_key=api_key)
        
        # Lấy các thông số từ config, sử dụng giá trị mặc định nếu thiếu
        self.model_name = config.get("name", "gemini-2.0-flash")
        self.temperature = config.get("temperature", 0.0)

    def generate(self, prompt):
        """
        Gửi prompt đến Gemini API và trả về chuỗi văn bản phản hồi.
        """
        try:
            generate_config = {
                'temperature': self.temperature,
                'max_output_tokens': self.model_config.get('max_tokens', 100)
            }
            
            # Gọi API theo chuẩn SDK mới
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generate_config
            )
            
            # Kiểm tra và trả về nội dung text
            if response and response.text:
                return response.text.strip()
            return ""
            
        except Exception as e:
            # Ghi log lỗi chi tiết để phục vụ debug khi chạy dataset lớn
            logging.error(f"Lỗi API Gemini tại model {self.model_name}: {e}")
            # Trả về chuỗi lỗi để hàm evaluate có thể bắt được (ví dụ: lỗi 429)
            return f"Error: {str(e)}"

    def score_plausibility(self, sample):
        """
        Quy trình chấm điểm: Tạo prompt -> Gọi LLM -> Trích xuất điểm số.
        """
        # self.create_prompt được kế thừa từ BaseLLMScorer
        prompt = self.create_prompt(sample)
        
        raw_response = self.generate(prompt)
        
        # self.extract_rating được kế thừa từ BaseLLMScorer (sử dụng Regex)
        rating = self.extract_rating(raw_response)
        
        return rating, raw_response