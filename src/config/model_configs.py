LLM_MODEL_CONFIGS = {

    "llama-70B": { 
        "name": "llama-3.3-70b-versatile", 
        "description": "Llama 3.3 70B ",
        "type": "groq",
        "max_tokens": 150, 
        "temperature": 0.0,
        "rpm_limit": 30 
    },

    "llama-instant": {
        "name": "llama-3.1-8b-instant",
        "description": "Llama 3.1 8B (Super Fast)",
        "type": "groq",
        "max_tokens": 150,
        "temperature": 0.0,
        "rpm_limit": 30
    },

    "qwen-32b": {
        "name": "qwen/qwen3-32b",
        "description": "Qwen 3 32B (Latest from Alibaba)",
        "type": "groq",
        "max_tokens": 200,
        "temperature": 0.0,
        "rpm_limit": 30
    }
}