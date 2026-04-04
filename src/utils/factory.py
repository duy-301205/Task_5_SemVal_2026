from src.config.model_configs import LLM_MODEL_CONFIGS
from src.models.groq_scorer import GroqLLMScorer

def create_llm_scorer(model_key, prompt_strategy="chain_of_thought", api_key=None):
    if model_key not in LLM_MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key: {model_key}")

    config = LLM_MODEL_CONFIGS[model_key]
    model_type = config.get("type")

    # 1. Khởi tạo cho Groq (Gemma 2)
    if model_type == "groq":
        return GroqLLMScorer(
            model_key=model_key,
            config=config,
            api_key=api_key,
            prompt_strategy=prompt_strategy
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")