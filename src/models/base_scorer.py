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

        if self.prompt_strategy == "improved":
            return PromptTemplate.improved(sample)

    def extract_rating(self, response: str) -> float:
        match = re.search(r'\b([1-5](?:\.[0-9]+)?)\b', response)
        return float(match.group(1)) if match else 3.0

    def score_plausibility(self, sample):

        raise NotImplementedError