import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.base_scorer import BaseLLMScorer
import numpy as np


class HuggingFaceLLMScorer(BaseLLMScorer):

    def __init__(self, model_key, config, prompt_strategy="basic"):
        super().__init__(model_key, config, prompt_strategy)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading model:", config["name"])

        self.tokenizer = AutoTokenizer.from_pretrained(config["name"])

        self.model = AutoModelForCausalLM.from_pretrained(
            config["name"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        self.model.eval()

    def generate(self, prompt):

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=self.config["temperature"],
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return text.strip()

    def score_plausibility(self, sample):

        prompt = self.create_prompt(sample)

        scores = []

        for _ in range(2):  # self-consistency
            response = self.generate(prompt)
            rating = self.extract_rating(response)
            scores.append(rating)

        return float(np.mean(scores)), response