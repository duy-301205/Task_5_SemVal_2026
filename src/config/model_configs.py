LLM_MODEL_CONFIGS = {
    # LỰA CHỌN SỐ 1: Ổn định nhất, Quota cao nhất
    "gemini-flash": {
        "name": "models/gemini-flash-latest", 
        "type": "gemini",
        "max_tokens": 512, 
        "temperature": 0.0 
    },
    
    # LỰA CHỌN SỐ 2: Siêu nhanh, Quota cực thoáng
    "gemini-flash-lite": {
        "name": "models/gemini-flash-lite-latest",
        "type": "gemini",
        "max_tokens": 400,
        "temperature": 0.0
    },

    # Chỉ dùng bản Pro khi thực sự cần độ chính xác cực cao cho các câu khó
    "gemini-pro": {
        "name": "models/gemini-pro-latest",
        "type": "gemini",
        "max_tokens": 800,
        "temperature": 0.0
    }
}