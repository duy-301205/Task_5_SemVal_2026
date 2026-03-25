LLM_MODEL_CONFIGS = {

    # 🔥 BEST BALANCE (RECOMMENDED)
    "gemini-flash": {
        "name": "gemini-2.5-flash",
        "description": "Gemini 2.5 Flash",
        "type": "gemini",
        "max_tokens": 10,
        "temperature": 0.0,
        "rpm_limit": 5
    },

    # ⚡ FAST + CHEAP
    "gemini-flash-lite": {
        "name": "gemini-2.0-flash-lite",
        "description": "Gemini 2.0 Flash Lite",
        "type": "gemini",
        "max_tokens": 10,
        "temperature": 0.0,
        "rpm_limit": 5
    },

    # 🧠 BEST ACCURACY (NCKH)
    "gemini-pro": {
        "name": "gemini-2.5-pro",
        "description": "Gemini 2.5 Pro",
        "type": "gemini",
        "max_tokens": 15,
        "temperature": 0.0,
        "rpm_limit": 5
    }
}