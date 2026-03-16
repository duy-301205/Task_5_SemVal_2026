import numpy as np
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import time
import json
import os

def evaluate_llm_scorer(scorer, samples, save_path="results/temp_results.json"):
    # 1. Tải kết quả đã lưu trước đó (Resume logic)
    processed_results = []
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            processed_results = json.load(f)
    
    # Tạo danh sách các ID đã xong để skip
    done_ids = {str(res['id']) for res in processed_results}
    print(f"🔄 Đã tìm thấy {len(done_ids)} mẫu đã hoàn thành. Đang chạy tiếp...")

    predictions = [res['prediction'] for res in processed_results]
    ground_truth = [float(res['ground_truth']) for res in processed_results if res['ground_truth'] is not None]
    
    pbar = tqdm(samples, desc=f"🚀 {scorer.model_key}")
    
    for sample in pbar:
        sample_id = str(sample['id'])
        
        # Nếu mẫu này đã chạy rồi thì bỏ qua
        if sample_id in done_ids:
            continue

        success = False
        retries = 0
        max_retries = 3

        while not success and retries < max_retries:
            try:
                # Gọi API
                pred, raw_res = scorer.score_plausibility(sample)
                
                # Nếu API trả về lỗi 429 (thường nằm trong raw_res nếu Duy dùng try-except ở Scorer)
                if isinstance(raw_res, str) and "429" in raw_res:
                    wait_time = 30 * (retries + 1)
                    print(f"\n⚠️ Bị chặn Quota câu {sample_id}. Ngủ {wait_time}s rồi thử lại...")
                    time.sleep(wait_time)
                    retries += 1
                    continue

                if isinstance(pred, (int, float)):
                    predictions.append(pred)
                    avg = sample.get("average")
                    ground_truth.append(float(avg) if avg is not None else 3.0)
                    
                    processed_results.append({
                        "id": sample["id"],
                        "prediction": pred,
                        "ground_truth": avg,
                        "raw_response": raw_res
                    })
                    
                    # Checkpoint: Lưu ngay lập tức
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(processed_results, f, indent=4, ensure_ascii=False)
                    
                    success = True
                    pbar.set_postfix({"id": sample_id, "score": pred})
                
                # Nghỉ giữa các request thành công (Duy nên để 8-10s cho an toàn)
                time.sleep(10) 

            except Exception as e:
                print(f"\n❌ Lỗi nghiêm trọng tại mẫu {sample_id}: {e}")
                time.sleep(20) # Ngủ lâu một chút nếu crash
                retries += 1

    # --- Tính toán Metrics ---
    preds_arr = np.array(predictions)
    gt_arr = np.array(ground_truth)

    if len(preds_arr) < 2 or np.unique(preds_arr).size == 1:
        return {"spearman": 0, "pearson": 0, "mae": 0, "msg": "Dữ liệu constant hoặc lỗi API toàn bộ"}

    spearman = spearmanr(preds_arr, gt_arr).correlation
    pearson = pearsonr(preds_arr, gt_arr)[0]
    mae = np.mean(np.abs(preds_arr - gt_arr))

    return {
        "spearman": spearman,
        "pearson": pearson,
        "mae": mae,
        "count": len(preds_arr)
    }