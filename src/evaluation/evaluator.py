import os
import json
import time
import random
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

def evaluate_llm_scorer(scorer, samples, save_path="results/temp_results.json"):
  
    processed_results = []
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                processed_results = json.load(f)
            print(f"🔄 Resume: Loaded {len(processed_results)} samples.")
        except Exception as e:
            print(f"⚠️ Checkpoint error: {e}")

    seen_ids = {str(r["id"]) for r in processed_results}
    pending_samples = [s for s in samples if str(s["id"]) not in seen_ids]

    # Xáo trộn để tránh bias theo cụm dữ liệu
    random.shuffle(pending_samples)

    print(f"🚀 Total: {len(samples)} | Done: {len(seen_ids)} | Left: {len(pending_samples)}")
    
    pbar = tqdm(pending_samples, desc=f"🤖 {scorer.model_key}")

    for sample in pbar:
        sample_id = str(sample["id"])

        try:
            gt_val = float(sample.get("average"))
            stdev_val = float(sample.get("stdev", 0.0)) 
        except (TypeError, ValueError):
            continue

        success = False
        retries = 0

        while not success and retries < 5:
            try:
                pred, raw_res = scorer.score_plausibility(sample)

                # --- KIỂM TRA LỖI NGHIÊM TRỌNG (DỪNG CHƯƠNG TRÌNH) ---
                if raw_res == "ERROR_FATAL" or pred == -1.0:
                    print(f"\n\n🛑 LỖI NGHIÊM TRỌNG: API của Groq gặp sự cố (Model die hoặc lỗi Auth).")
                    print(f"📍 Mẫu bị lỗi ID: {sample_id}")
                    print(f"👉 Vui lòng kiểm tra lại cấu hình model hoặc API Key trong .env")
                    # Dừng toàn bộ chương trình ngay lập tức
                    raise SystemExit

                # --- KIỂM TRA RATE LIMIT (THỬ LẠI) ---
                if raw_res == "ERROR_429":
                    wait_time = 20 * (retries + 1)
                    print(f"\n⚠️ GROQ LIMIT -> Sleep {wait_time}s")
                    time.sleep(wait_time)
                    retries += 1
                    continue

                # Nếu chạy đến đây nghĩa là lấy điểm thành công
                pred = float(np.clip(pred, 1.0, 5.0))

                # Lưu kết quả
                processed_results.append({
                    "id": sample["id"],
                    "prediction": pred,
                    "ground_truth": gt_val,
                    "stdev": stdev_val,
                    "raw_response": raw_res
                })

                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_results, f, indent=2, ensure_ascii=False)

                success = True
                pbar.set_postfix({"id": sample_id, "score": f"{pred:.2f}"})
                
                # Tốc độ an toàn cho Groq bản Free
                time.sleep(2.0) 

            except SystemExit:
                raise # Đẩy lệnh dừng chương trình ra ngoài
            except Exception as e:
                wait_time = min(60, 5 * (2**retries))
                print(f"\n❌ Network Error -> Sleep {wait_time}s | {e}")
                time.sleep(wait_time)
                retries += 1

    return calculate_metrics(processed_results)

def calculate_metrics(results):
    """
    Tính toán các chỉ số metrics cuối cùng.
    """
    seen = set()
    valid_data = []

    for r in results:
        if r["id"] in seen: continue
        # Chỉ tính toán trên các kết quả không phải lỗi (khác -1.0)
        if isinstance(r.get("prediction"), (int, float)) and r["prediction"] != -1.0:
            valid_data.append(r)
            seen.add(r["id"])

    if len(valid_data) < 2:
        return {"msg": "Not enough valid data to compute metrics."}

    valid_data = sorted(valid_data, key=lambda x: x["id"])

    preds = np.array([r["prediction"] for r in valid_data])
    gts = np.array([r["ground_truth"] for r in valid_data])
    stdevs = np.array([r.get("stdev", 0.0) for r in valid_data])

    mae = np.mean(np.abs(preds - gts))
    rmse = np.sqrt(np.mean((preds - gts) ** 2))

    # Acc.std: |pred - gt| <= stdev
    acc_std_list = np.abs(preds - gts) <= stdevs
    acc_std = np.mean(acc_std_list)

    metrics = {
        "count": len(valid_data),
        "mae": float(mae),
        "rmse": float(rmse),
        "acc_std": float(acc_std),
    }

    if np.unique(preds).size > 1:
        metrics["spearman"] = float(spearmanr(preds, gts).correlation)
        metrics["pearson"] = float(pearsonr(preds, gts)[0])
    else:
        metrics["warning"] = "Constant predictions detected"

    return metrics