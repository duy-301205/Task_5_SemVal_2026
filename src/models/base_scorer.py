import re
import numpy as np
from src.prompts.prompt_templates import PromptTemplate

class BaseLLMScorer:
    def __init__(self, model_key, config, prompt_strategy="cot"):
        self.model_key = model_key
        self.config = config
        self.prompt_strategy = prompt_strategy

    def create_prompt(self, sample):
        """
        Chỉ giữ lại 3 chiến thuật cải tiến mạnh nhất cho SemEval Task 5.
        """
        if self.prompt_strategy == "chain_of_thought":
            return PromptTemplate.chain_of_thought(sample)
        if self.prompt_strategy == "one_shot":
            return PromptTemplate.one_shot(sample)
        if self.prompt_strategy == "few_shot":
            return PromptTemplate.few_shot(sample)
        
        # Mặc định dùng Chain of Thought nếu không chỉ định rõ
        return PromptTemplate.chain_of_thought(sample)

    def extract_rating(self, response: str) -> float:
        """
        Trích xuất điểm số từ phản hồi của LLM. 
        Nếu nhận được tín hiệu lỗi từ Scorer, trả về -1.0 để dừng chương trình.
        """
        if not response or "ERROR_FATAL" in response:
            return -1.0

        # Chuẩn hóa chuỗi
        response = response.strip().replace(',', '.')

        # 1. Tìm theo nhãn (Ví dụ: 'Rating: 4.5' hoặc 'Score: 2.0')
        label_match = re.search(
            r'(?:rating|score|value|plausibility)\s*[:\-]?\s*([1-5](?:\.\d+)?)',
            response,
            re.IGNORECASE
        )
        if label_match:
            return float(label_match.group(1))

        # 2. Tìm số cuối cùng trong văn bản ( AI thường chốt kết quả ở cuối)
        all_numbers = re.findall(r'(?<![\d\.])([1-5](?:\.\d+)?)(?![\d\.])', response)
        
        if all_numbers:
            val = float(all_numbers[-1])
            return max(1.0, min(5.0, val))

        # 3. Nếu không tìm thấy số hợp lệ nào, coi như lỗi định dạng phản hồi
        return -1.0

    def score_plausibility(self, sample):
        """Hàm này sẽ được override bởi GroqLLMScorer hoặc GeminiLLMScorer"""
        raise NotImplementedError