import json
import os
from dotenv import load_dotenv

from src.parser.ambistory_parser import AmbiStoryParser
from src.utils.factory import create_llm_scorer
from src.evaluation.evaluator import evaluate_llm_scorer

DATA_PATH = "data/dev.json"

def main():
    # 1. Load biến môi trường
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key is None:
        raise ValueError("GEMINI_API_KEY not found. Please set it in .env file")

    # 2. Đọc và parse dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    parser = AmbiStoryParser(data)
    
    full_samples = parser.get_samples()
    samples = full_samples[:10]
    
    # 3. Cấu hình thí nghiệm
    model_key = "gemini-flash" 
    strategy = "semeval_official"
    
    # Tạo đường dẫn lưu kết quả riêng biệt cho từng model/strategy
    save_dir = "results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f"{save_dir}/{model_key}_{strategy}_results.json"

    print(f"=== Bắt đầu thí nghiệm ===")
    print(f"Model: {model_key}")
    print(f"Strategy: {strategy}")
    print(f"Dữ liệu lưu tại: {save_path}")
    print(f"Số lượng mẫu: {len(samples)}")
    print("---------------------------")

    # 4. Tạo scorer
    scorer = create_llm_scorer(
        model_key=model_key,
        prompt_strategy=strategy,
        api_key=api_key
    )

    # 5. Evaluation (Thêm save_path để kích hoạt cơ chế Resume)
    results = evaluate_llm_scorer(scorer, samples, save_path=save_path)

    print("\n" + "="*30)
    print("KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric.upper()}: {value:.4f}")
        else:
            print(f"{metric.upper()}: {value}")
    print("="*30)

if __name__ == "__main__":
    main()