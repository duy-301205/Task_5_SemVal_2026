import re
import numpy as np
from src.prompts.prompt_templates import PromptTemplate

class BaseLLMScorer:
    def __init__(self, model_key, config, prompt_strategy="basic"):
        self.model_key = model_key
        self.config = config
        self.prompt_strategy = prompt_strategy

    def create_prompt(self, sample):
        if self.prompt_strategy == "basic":
            return PromptTemplate.basic(sample)
        if self.prompt_strategy == "criteria":
            return PromptTemplate.criteria(sample)
        if self.prompt_strategy == "semeval_official":
            return PromptTemplate.semeval_official(sample)
        if self.prompt_strategy == "improved":
            return PromptTemplate.improved(sample)
        return PromptTemplate.basic(sample)

    def extract_rating(self, response: str) -> float:
        if not response:
            return 3.0

        # Clean up string
        response = response.strip().replace(',', '.') # Đổi dấu phẩy thành dấu chấm nếu có

        # 1. Ưu tiên dạng có label (Rating: 4.5)
        label_match = re.search(
            r'(?:rating|score|value|plausibility)\s*[:\-]?\s*([1-5](?:\.\d+)?)',
            response,
            re.IGNORECASE
        )
        if label_match:
            return float(label_match.group(1))

        # 2. Tìm tất cả các số từ 1.0 đến 5.0 trong văn bản
        # Thay \b bằng các khoảng trắng hoặc biên linh hoạt hơn
        all_numbers = re.findall(r'(?<![\d\.])([1-5](?:\.\d+)?)(?![\d\.])', response)
        
        if all_numbers:
            # Lấy số cuối cùng vì AI thường chốt hạ ở cuối câu
            val = float(all_numbers[-1])
            return max(1.0, min(5.0, val))

        # 3. Fallback trung tính
        return 3.0

    def score_plausibility(self, sample):
        raise NotImplementedError