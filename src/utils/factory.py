from src.config.model_configs import LLM_MODEL_CONFIGS
from src.models.hf_scorer import HuggingFaceLLMScorer
from src.models.gemini_scorer import GeminiLLMScorer


def create_llm_scorer(model_key, prompt_strategy="basic", api_key=None):

    if model_key not in LLM_MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key: {model_key}")

    config = LLM_MODEL_CONFIGS[model_key]
    model_type = config.get("type")

    print(f"[INFO] Initializing model={model_key} | type={model_type} | prompt={prompt_strategy}")

    if model_type == "huggingface":
        return HuggingFaceLLMScorer(
            model_key=model_key,
            config=config,
            prompt_strategy=prompt_strategy
        )

    elif model_type == "gemini":
        if not api_key:
            raise ValueError("Gemini API key is required")

        return GeminiLLMScorer(
            model_key=model_key,
            config=config,
            api_key=api_key,
            prompt_strategy=prompt_strategy
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")