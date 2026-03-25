import os
import json
import time
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

def evaluate_llm_scorer(scorer, samples, save_path="results/temp_results.json"):
    import os, json, time, random
    import numpy as np
    from tqdm import tqdm
    from scipy.stats import spearmanr, pearsonr

    # ===============================
    # LOAD CHECKPOINT
    # ===============================
    processed_results = []

    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                processed_results = json.load(f)
            print(f"🔄 Resume: loaded {len(processed_results)} samples")
        except:
            print("⚠️ Cannot read checkpoint → restart")

    # tránh duplicate
    seen_ids = {str(r["id"]) for r in processed_results}

    pending_samples = [s for s in samples if str(s["id"]) not in seen_ids]

    random.shuffle(pending_samples)

    print(f"🚀 Total: {len(samples)} | Done: {len(seen_ids)} | Left: {len(pending_samples)}")

    pbar = tqdm(pending_samples, desc=f"🤖 {scorer.model_key}")

    # ===============================
    # MAIN LOOP
    # ===============================
    for sample in pbar:
        sample_id = str(sample["id"])

        # ===== SAFE GT =====
        try:
            gt_val = float(sample.get("average"))
        except:
            continue

        success = False
        retries = 0

        while not success and retries < 5:
            try:
                pred, raw_res = scorer.score_plausibility(sample)

                # ===== CLIP =====
                pred = float(np.clip(pred, 1.0, 5.0))

                # ===== CHECK QUOTA ONLY =====
                res_str = str(raw_res).lower()
                if "429" in res_str or "exhausted" in res_str:
                    wait_time = 60
                    print(f"\n⚠️ QUOTA → sleep {wait_time}s")
                    time.sleep(wait_time)
                    retries += 1
                    continue

                # ===== SAVE RESULT =====
                processed_results.append({
                    "id": sample["id"],
                    "prediction": pred,
                    "ground_truth": gt_val,
                    "raw_response": raw_res
                })

                # save ngay (anti-crash)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_results, f, indent=2, ensure_ascii=False)

                success = True
                pbar.set_postfix({"id": sample_id, "score": pred})

                # ===== RATE LIMIT SAFE =====
                time.sleep(12)

            except Exception as e:
                wait_time = min(60, 5 * (2 ** retries))
                print(f"\n❌ Error → sleep {wait_time}s | {e}")
                time.sleep(wait_time)
                retries += 1

    # ===============================
    # METRICS (FIXED)
    # ===============================

    # remove duplicate + invalid
    seen = set()
    valid_data = []

    for r in processed_results:
        if r["id"] in seen:
            continue
        if isinstance(r.get("prediction"), (int, float)):
            valid_data.append(r)
            seen.add(r["id"])

    if len(valid_data) < 2:
        return {"msg": "Not enough valid data"}

    # SORT để align chuẩn
    valid_data = sorted(valid_data, key=lambda x: x["id"])

    preds = np.array([r["prediction"] for r in valid_data])
    gts = np.array([r["ground_truth"] for r in valid_data])

    metrics = {
        "count": len(valid_data),
        "mae": float(np.mean(np.abs(preds - gts))),
        "rmse": float(np.sqrt(np.mean((preds - gts) ** 2))),
    }

    # tránh constant prediction
    if np.unique(preds).size > 1:
        metrics["spearman"] = float(spearmanr(preds, gts).correlation)
        metrics["pearson"] = float(pearsonr(preds, gts)[0])
    else:
        metrics["warning"] = "Constant predictions"

    return metrics