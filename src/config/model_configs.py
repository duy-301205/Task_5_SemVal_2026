LLM_MODEL_CONFIGS = {
    # LỰA CHỌN SỐ 1: Mạnh mẽ và hiện đại nhất hiện nay (Gemini 2.5)
    "gemini-flash": {
        "name": "models/gemini-2.5-flash", 
        "type": "gemini",
        "max_tokens": 512, 
        "temperature": 0.0 
    },
    
    # LỰA CHỌN SỐ 2: Tối ưu tốc độ với phiên bản 2.0 Flash-Lite
    "gemini-flash-lite": {
        "name": "models/gemini-2.0-flash-lite",
        "type": "gemini",
        "max_tokens": 400,
        "temperature": 0.0
    },

    # LỰA CHỌN SỐ 3: Đỉnh cao lập luận (Sử dụng 2.5 Pro cho độ chính xác NCKH)
    "gemini-pro": {
        "name": "models/gemini-2.5-pro",
        "type": "gemini",
        "max_tokens": 800,
        "temperature": 0.0
    }
}