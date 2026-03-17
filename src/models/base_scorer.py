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
        """
        Trích xuất điểm số từ phản hồi của LLM.
        Hỗ trợ cả số nguyên và số thập phân (ví dụ: 3.5, 4.2).
        """
        if not response:
            return 3.0

        # 1. Thử tìm định dạng "Rating: X" hoặc "Score: X" trước để tránh lấy nhầm số trong văn bản
        label_match = re.search(r'(?:rating|score|value)[:\s]+([1-5](?:\.[0-9]+)?)', response, re.IGNORECASE)
        if label_match:
            return float(label_match.group(1))

        # 2. Nếu không thấy label, tìm số đầu tiên xuất hiện trong dải 1-5
        # Regex này bắt được: "5", "4.5", "3.25", v.v.
        match = re.search(r'([1-5](?:\.[0-9]+)?)', response)
        
        if match:
            val = float(match.group(1))
            # Đảm bảo giá trị không vượt quá dải [1, 5]
            return max(1.0, min(5.0, val))
            
        # 3. Trả về điểm trung tính nếu thất bại hoàn toàn
        return 3.0

    def score_plausibility(self, sample):
        raise NotImplementedError