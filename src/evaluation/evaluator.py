import os
import json
import time
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

def evaluate_llm_scorer(scorer, samples, save_path="results/temp_results.json"):
    # 1. Tải kết quả cũ (Resume)
    processed_results = []
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                processed_results = json.load(f)
        except Exception as e:
            print(f"⚠️ Lỗi đọc file cũ: {e}")

    done_ids = {str(res['id']) for res in processed_results}
    print(f"🔄 Đã xong: {len(done_ids)}/{len(samples)}. Đang chạy tiếp...")

    pbar = tqdm(samples, desc=f"🚀 {scorer.model_key}")
    
    for sample in pbar:
        sample_id = str(sample['id'])
        if sample_id in done_ids:
            continue

        avg = sample.get("average")
        if avg is None or avg == "(???)":
            continue
        gt_val = float(avg)

        success = False
        retries = 0
        max_retries = 5 

        while not success and retries < max_retries:
            try:
                # GỌI API
                pred, raw_res = scorer.score_plausibility(sample)
                
                # Nếu raw_res chứa chữ "Error" hoặc "failed" hoặc "429"
                is_error = isinstance(raw_res, str) and any(x in raw_res.lower() for x in ["error", "429", "failed", "11002"])
                
                if is_error:
                    wait_time = (30 * retries) + 20 
                    print(f"\n⚠️ Lỗi kết nối/API tại câu {sample_id}: {raw_res}. Thử lại sau {wait_time}s...")
                    time.sleep(wait_time)
                    retries += 1
                    continue 

                # TỐI ƯU 2: Chỉ lưu khi pred là số thực thụ và không phải là lỗi
                if isinstance(pred, (int, float)) and 1 <= pred <= 5:
                    processed_results.append({
                        "id": sample["id"],
                        "prediction": pred,
                        "ground_truth": gt_val,
                        "raw_response": raw_res
                    })
                    
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(processed_results, f, indent=4, ensure_ascii=False)
                    
                    success = True
                    pbar.set_postfix({"id": sample_id, "score": pred})
                    time.sleep(15) 
                else:
                    print(f"\n⚠️ Không trích xuất được điểm hợp lệ câu {sample_id}. Thử lại {retries+1}/{max_retries}")
                    retries += 1
                    time.sleep(2)

            except Exception as e:
                # Lỗi crash hệ thống (ví dụ mất mạng hoàn toàn)
                print(f"\n❌ Lỗi hệ thống tại mẫu {sample_id}: {e}")
                time.sleep(30) 
                retries += 1

    # --- Tính toán Metrics ---
    valid_data = [r for r in processed_results if "error" not in str(r.get('raw_response', '')).lower()]
    
    if len(valid_data) < 2:
        return {"msg": "Không đủ dữ liệu sạch"}

    preds_arr = np.array([r['prediction'] for r in valid_data])
    gt_arr = np.array([r['ground_truth'] for r in valid_data])

    if np.unique(preds_arr).size == 1:
        return {"mae": np.mean(np.abs(preds_arr - gt_arr)), "msg": "Dự đoán bị hằng số (Constant)"}

    return {
        "spearman": spearmanr(preds_arr, gt_arr).correlation,
        "pearson": pearsonr(preds_arr, gt_arr)[0],
        "mae": np.mean(np.abs(preds_arr - gt_arr)),
        "count": len(valid_data)
    }