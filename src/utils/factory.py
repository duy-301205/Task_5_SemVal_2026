from src.config.model_configs import LLM_MODEL_CONFIGS
from src.models.hf_scorer import HuggingFaceLLMScorer
from src.models.gemini_scorer import GeminiLLMScorer


def create_llm_scorer(model_key, prompt_strategy="basic", api_key=None):
    config = LLM_MODEL_CONFIGS[model_key]

    if config["type"] == "huggingface":
        return HuggingFaceLLMScorer(
            model_key=model_key,
            config=config,
            prompt_strategy=prompt_strategy
        )

    if config["type"] == "gemini":
        return GeminiLLMScorer(
            model_key=model_key,
            config=config,
            api_key=api_key,
            prompt_strategy=prompt_strategy
        )