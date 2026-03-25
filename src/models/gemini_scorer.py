from google import genai
from .base_scorer import BaseLLMScorer
from typing import Optional, Dict, Tuple
import numpy as np
import time
from threading import Semaphore

class GeminiLLMScorer(BaseLLMScorer):

    def __init__(self, model_key: str, config, prompt_strategy: str = 'basic',
                 api_key: Optional[str] = None):
        super().__init__(model_key, config, prompt_strategy)

        if not api_key:
            raise ValueError("Gemini API Key is required.")

        self.client = genai.Client(api_key=api_key)
        self.semaphore = Semaphore(5)

        print(f"Initialized {self.config['description']}")

    def generate_response(self, prompt: str) -> str:
        for _ in range(3):
            try:
                with self.semaphore:
                    response = self.client.models.generate_content(
                        model=self.config['name'],
                        contents=prompt,
                        config={
                            "max_output_tokens": self.config.get('max_tokens', 40),
                            "temperature": self.config.get('temperature', 0.1),
                            "system_instruction": "Return ONLY a number from 1 to 5."
                        }
                    )
                return response.text.strip() if response.text else "3"

            except Exception as e:
                if "429" in str(e):
                    time.sleep(5)
                else:
                    return "3"

        return "3"

    def score_plausibility(self, sample: Dict) -> Tuple[float, str]:
        prompt = self.create_prompt(sample)
        scores = []

        for _ in range(2):  # self-consistency
            res = self.generate_response(prompt)
            scores.append(self.extract_rating(res))

        return float(np.mean(scores)), res