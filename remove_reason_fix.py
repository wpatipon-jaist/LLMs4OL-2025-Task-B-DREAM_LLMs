import json
import argparse
from pathlib import Path
import os

AVAILABLE_DATASETS = ["MatOnto", "OBI", "SWEET"]
RESULT_DIR = Path("result_with_reason")
RESULT_FOR_SUBMIT = Path("result_with_reason_for_submit")
MODEL_NAME = ["gpt-4o", "gemini-2.5-pro", "claude-sonnet-4-20250514", "deepseek-chat"]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=AVAILABLE_DATASETS + ["all"],
        help="Dataset to process or 'all' to process all datasets",
    )

    parser.add_argument(
        "--model",
        choices=MODEL_NAME + ["all"],
        help="Dataset to process or 'all' to process all datasets",
    )

    args = parser.parse_args()

    datasets_to_process = AVAILABLE_DATASETS if args.dataset == "all" else [args.dataset]
    models = MODEL_NAME if args.model == "all" else [args.model]

    print(f"Processing datasets: {datasets_to_process} with models: {models}")

    for dataset_name in datasets_to_process:
        for model_name in models:
            filename = RESULT_DIR.joinpath(model_name).joinpath(f"{dataset_name.lower()}.json")
            if not filename.exists():
                print(f"File {filename} does not exist. Skipping...")
                continue
            
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)

            print(f"Processing {dataset_name} with {model_name}...")

            result_filename = RESULT_FOR_SUBMIT.joinpath(model_name).joinpath(f"{dataset_name.lower()}.json")
            os.makedirs(result_filename.parent, exist_ok=True)
            
            formatted_results = [{"id": item["id"], "types": item["types"]} for item in data]
            with open(result_filename, "w", encoding="utf-8") as f:
                json.dump(formatted_results, f, indent=2)

if __name__ == "__main__":
    main()