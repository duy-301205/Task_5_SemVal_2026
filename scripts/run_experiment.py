import json
import os
import random
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

from src.parser.ambistory_parser import AmbiStoryParser
from src.utils.factory import create_llm_scorer
from src.evaluation.evaluator import evaluate_llm_scorer

DATA_PATH = "data/train.json"
MAX_SAMPLES = None  # set = 10 để debug

def main():
    # ===== SEED =====
    random.seed(42)
    np.random.seed(42)

    # ===== ENV =====
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY")

    # ===== LOAD DATA =====
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    parser = AmbiStoryParser(data)
    full_samples = parser.get_samples()

    samples = full_samples if MAX_SAMPLES is None else full_samples[:MAX_SAMPLES]

    # ===== CONFIG =====
    model_key = "gemini-flash"
    strategy = "semeval_official"

    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    save_path = f"{save_dir}/{model_key}_{strategy}.json"

    print("\n=== EXPERIMENT START ===")
    print({
        "model": model_key,
        "strategy": strategy,
        "samples": len(samples),
        "save": save_path
    })
    print("========================\n")

    # ===== SCORER =====
    scorer = create_llm_scorer(
        model_key=model_key,
        prompt_strategy=strategy,
        api_key=api_key
    )

    # ===== RUN =====
    try:
        results = evaluate_llm_scorer(
            scorer,
            samples,
            save_path=save_path
        )
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return

    # ===== PRINT =====
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)

    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k.upper()}: {v:.4f}")
        else:
            print(f"{k.upper()}: {v}")

    print("="*40)


if __name__ == "__main__":
    main()