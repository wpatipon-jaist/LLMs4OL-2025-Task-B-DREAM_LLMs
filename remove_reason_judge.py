import json
import argparse
from pathlib import Path
import os

AVAILABLE_DATASETS = ["MatOnto", "OBI", "SWEET"]
RESULT_DIR = Path("results_judge")
RESULT_FOR_SUBMIT = Path("results_for_submit_judge")
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

    for model_name in models:
        model_result_dir = RESULT_DIR.joinpath(model_name)
        if not model_result_dir.exists() or not model_result_dir.is_dir():
            print(f"Directory {model_result_dir} does not exist. Skipping...")
            continue
        
        # Get all JSON files in the directory
        json_files = list(model_result_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {model_result_dir}. Skipping...")
            continue
        
        print(f"Processing {len(json_files)} files for model {model_name}...")
        
        # Create output directory
        output_dir = RESULT_FOR_SUBMIT.joinpath(model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        for json_file in json_files:
            print(f"Processing file: {json_file.name}")
            
            with open(json_file, "r") as f:
                data = json.load(f)
            
            # Determine output filename
            output_filename = output_dir.joinpath(f"{json_file.stem}_{model_name}.json")
            
            # Transform data
            formatted_results = [{"id": item["id"], "types": item["types"]} for item in data]
            
            with open(output_filename, "w") as f:
                json.dump(formatted_results, f, indent=2)

if __name__ == "__main__":
    main()