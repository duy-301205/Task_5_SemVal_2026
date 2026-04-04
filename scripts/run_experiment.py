import json
import os
import random
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

from src.parser.ambistory_parser import AmbiStoryParser
from src.utils.factory import create_llm_scorer
from src.evaluation.evaluator import evaluate_llm_scorer

DATA_PATH = "data/dev.json"
MAX_SAMPLES = None  

def main():
    random.seed(42)
    np.random.seed(42)

    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ Không tìm thấy GROQ_API_KEY hoặc GEMINI_API_KEY trong file .env")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu tại: {DATA_PATH}")

    print(f"📂 Đang tải dữ liệu từ {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    parser = AmbiStoryParser(data)
    full_samples = parser.get_samples()

    samples = full_samples if MAX_SAMPLES is None else full_samples[:MAX_SAMPLES]

    model_key = "llama-70B" 
    strategy = "one_shot" 

    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    save_path = f"{save_dir}/{model_key}_{strategy}.json"

    print("\n" + "="*30)
    print("🚀 BẮT ĐẦU THỰC NGHIỆM")
    print(f"🤖 Model   : {model_key}")
    print(f"📝 Strategy: {strategy}")
    print(f"📊 Samples : {len(samples)}")
    print(f"💾 Save to : {save_path}")
    print("="*30 + "\n")

    # ===== 5. KHỞI TẠO SCORER (FACTORY) =====
    scorer = create_llm_scorer(
        model_key=model_key,
        prompt_strategy=strategy,
        api_key=api_key
    )

    # ===== 6. CHẠY ĐÁNH GIÁ (EVALUATOR) =====
    try:
        # Hàm này sẽ xử lý việc gọi API, lưu checkpoint và tính MAE, RMSE, Spearman, Pearson, Acc.std
        results = evaluate_llm_scorer(
            scorer,
            samples,
            save_path=save_path
        )
    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng trong quá trình chạy: {e}")
        import traceback
        traceback.print_exc()
        return

    # ===== 7. IN KẾT QUẢ CUỐI CÙNG =====
    print("\n" + "⭐"*20)
    print(" KẾT QUẢ THỰC NGHIỆM CUỐI CÙNG")
    print("⭐"*20)

    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k.upper():<10}: {v:.4f}")
        else:
            print(f"{k.upper():<10}: {v}")

    print("="*40)
    print(f"✅ Đã lưu kết quả chi tiết tại: {save_path}")


if __name__ == "__main__":
    main()