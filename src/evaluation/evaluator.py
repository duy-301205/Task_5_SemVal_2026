import os
import json
import time
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

def evaluate_llm_scorer(scorer, samples, save_path="results/temp_results.json"):
    """
    Hàm đánh giá LLM Scorer với cơ chế chống lỗi Quota 429 (Rate Limit).
    Phù hợp cho Duy chạy NCKH với tài khoản Gemini Free (5 RPM).
    """
    
    # 1. Tải kết quả cũ để chạy tiếp (Resume)
    processed_results = []
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                processed_results = json.load(f)
            print(f"🔄 Đã tìm thấy file cũ. Đang tải {len(processed_results)} mẫu đã xong...")
        except Exception as e:
            print(f"⚠️ Không thể đọc file cũ, bắt đầu mới hoàn toàn: {e}")

    done_ids = {str(res['id']) for res in processed_results}
    
    # Lọc danh sách samples chưa chạy
    pending_samples = [s for s in samples if str(s['id']) not in done_ids]
    
    print(f"🚀 Tổng: {len(samples)} | Đã xong: {len(done_ids)} | Còn lại: {len(pending_samples)}")
    print(f"ℹ️ Giới hạn hiện tại: 5 RPM (Yêu cầu nghỉ ít nhất 13s/câu)")

    pbar = tqdm(pending_samples, desc=f"🤖 {scorer.model_key}")
    
    for sample in pbar:
        sample_id = str(sample['id'])
        
        # Lấy nhãn trung bình (Ground Truth)
        avg = sample.get("average")
        if avg is None or avg == "(???)":
            continue
        gt_val = float(avg)

        success = False
        retries = 0
        max_retries = 5 

        while not success and retries < max_retries:
            try:
                # GỌI API QUA SCORER
                pred, raw_res = scorer.score_plausibility(sample)
                
                # Ép kiểu phản hồi về string để kiểm tra lỗi
                res_str = str(raw_res).lower()
                
                # Kiểm tra các dấu hiệu bị chặn Quota hoặc lỗi API
                is_error = any(x in res_str for x in ["error", "429", "failed", "11002", "exhausted", "permission_denied"])
                
                if is_error:
                    # NẾU LỖI QUOTA (429): Nghỉ hẳn 65 giây để reset "cửa sổ" 1 phút của Google
                    wait_time = 65 if ("429" in res_str or "exhausted" in res_str) else (30 * retries + 20)
                    print(f"\n⚠️ [QUOTA ALERT] Tại câu {sample_id}. Đang nghỉ {wait_time}s...")
                    time.sleep(wait_time)
                    retries += 1
                    continue 

                # KIỂM TRA ĐIỂM HỢP LỆ (1.0 - 5.0)
                if isinstance(pred, (int, float)) and 1 <= pred <= 5:
                    processed_results.append({
                        "id": sample["id"],
                        "prediction": pred,
                        "ground_truth": gt_val,
                        "raw_response": raw_res
                    })
                    
                    # Lưu file ngay lập tức sau mỗi câu (Phòng trường hợp crash)
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(processed_results, f, indent=4, ensure_ascii=False)
                    
                    success = True
                    pbar.set_postfix({"id": sample_id, "score": pred})
                    
                    # NGHỈ CHIẾN THUẬT: 14 giây để đảm bảo < 5 requests/phút (60/14 ~ 4.2 RPM)
                    time.sleep(14) 
                else:
                    print(f"\n⚠️ Model trả về điểm không hợp lệ ({pred}) tại {sample_id}. Thử lại...")
                    retries += 1
                    time.sleep(5)

            except Exception as e:
                # Lỗi hệ thống (Mất mạng, thư viện crash...)
                print(f"\n❌ Lỗi hệ thống nghiêm trọng: {e}")
                time.sleep(40) 
                retries += 1

    # --- TÍNH TOÁN METRICS CUỐI CÙNG ---
    # Lọc lấy các kết quả có prediction là số
    valid_data = [r for r in processed_results if isinstance(r.get('prediction'), (int, float))]
    
    if len(valid_data) < 2:
        return {"msg": "Không đủ dữ liệu sạch để tính Metrics (cần ít nhất 2 mẫu)"}

    preds_arr = np.array([r['prediction'] for r in valid_data])
    gt_arr = np.array([r['ground_truth'] for r in valid_data])

    metrics = {
        "mae": np.mean(np.abs(preds_arr - gt_arr)),
        "rmse": np.sqrt(np.mean((preds_arr - gt_arr)**2)), # Thêm RMSE cho NCKH
        "count": len(valid_data)
    }

    # Spearman và Pearson chỉ tính được khi kết quả không phải là hằng số
    if np.unique(preds_arr).size > 1:
        metrics["spearman"] = spearmanr(preds_arr, gt_arr).correlation
        metrics["pearson"] = pearsonr(preds_arr, gt_arr)[0]
    else:
        metrics["msg"] = "Cảnh báo: Tất cả các dự đoán đều giống hệt nhau (Constant)."

    return metrics