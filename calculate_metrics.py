import json
import numpy as np
from scipy.stats import spearmanr, pearsonr

def calculate_metrics_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file tại: {file_path}")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    seen = set()
    valid_data = []

    for r in results:
        if r["id"] in seen: 
            continue
            
        pred_val = r.get("prediction")
        if isinstance(pred_val, (int, float)) and pred_val != -1.0:
            if r.get("ground_truth") is not None:
                valid_data.append(r)
                seen.add(r["id"])

    if len(valid_data) < 2:
        print("⚠️ Không đủ dữ liệu hợp lệ (cần ít nhất 2 mẫu) để tính toán metrics.")
        return

    valid_data = sorted(valid_data, key=lambda x: x["id"])

    preds = np.array([float(r["prediction"]) for r in valid_data])
    gts = np.array([float(r["ground_truth"]) for r in valid_data])
    stdevs = np.array([float(r.get("stdev", 0.0)) for r in valid_data])

    mae = np.mean(np.abs(preds - gts))
    rmse = np.sqrt(np.mean((preds - gts) ** 2))

    acc_std_list = np.abs(preds - gts) <= stdevs
    acc_std = np.mean(acc_std_list)

    metrics = {
        "count": len(valid_data),
        "mae": float(mae),
        "rmse": float(rmse),
        "acc_std": float(acc_std * 100), 
    }

    if np.unique(preds).size > 1:
        metrics["spearman"] = float(spearmanr(preds, gts).correlation)
        metrics["pearson"] = float(pearsonr(preds, gts)[0])
    else:
        metrics["warning"] = "Constant predictions detected (AI trả về cùng một mức điểm cho mọi câu)"

    print(f"\n📊 KẾT QUẢ PHÂN TÍCH (N = {metrics['count']})")
    print("="*40)
    print(f"🔹 Spearman : {metrics.get('spearman', 0.0):.4f}")
    print(f"🔹 Acc.std  : {metrics['acc_std']:.2f}%")
    print(f"🔹 MAE       : {metrics['mae']:.4f}")
    print(f"🔹 RMSE      : {metrics['rmse']:.4f}")
    if "warning" in metrics:
        print(f"⚠️ Warning  : {metrics['warning']}")
    print("="*40)

    return metrics

if __name__ == "__main__":
    import os
    FILE_NAME = "results/llama-70B_one_shot.json"
    calculate_metrics_from_file(FILE_NAME)