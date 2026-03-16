import json
import os
from dotenv import load_dotenv

from src.parser.ambistory_parser import AmbiStoryParser
from src.utils.factory import create_llm_scorer
from src.evaluation.evaluator import evaluate_llm_scorer


DATA_PATH = "data/dev.json"


def main():

    # load biến môi trường từ .env
    load_dotenv()

    # lấy API key Gemini
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key is None:
        raise ValueError("GEMINI_API_KEY not found. Please set it in .env file")

    # đọc dataset
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # parse dataset
    parser = AmbiStoryParser(data)
    samples = parser.get_samples()[:50]

    # tạo scorer
    scorer = create_llm_scorer(
        model_key="gemini-flash",
        prompt_strategy="criteria",
        api_key=api_key
    )

    # evaluation
    results = evaluate_llm_scorer(scorer, samples)

    print("\nEvaluation results:")
    print(results)


if __name__ == "__main__":
    main()