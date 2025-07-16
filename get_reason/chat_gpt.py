import argparse
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
import os
import instructor
from pydantic import BaseModel
import json
from tqdm import tqdm
import csv

load_dotenv()
api_key = os.environ["OPEN_AI_API_KEY"] 

AVAILABLE_DATASETS = ["MatOnto", "OBI", "SWEET"]
OUTPUT_DIR = Path("../need_reason_data")
RESULT_DIR = Path("../result_with_reason")
MODEL_NAME = "gpt-4o"

client = instructor.from_openai(OpenAI(api_key=api_key))

class TermTypingWithReason(BaseModel):
    id: str
    types: list[str]
    reason: str

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        choices=AVAILABLE_DATASETS + ["all"],
        help="Dataset to process or 'all' to process all datasets",
    )

    args = parser.parse_args()

    datasets_to_process = AVAILABLE_DATASETS if args.dataset == "all" else [args.dataset]

    for dataset_name in datasets_to_process:
        filename = OUTPUT_DIR.joinpath(MODEL_NAME).joinpath(f"{dataset_name.lower()}.csv")
        prompt_filename = OUTPUT_DIR.joinpath(MODEL_NAME).joinpath(f"{dataset_name.lower()}_prompt.json")

        with open(prompt_filename, encoding="utf-8") as f:
            data = json.load(f)
            prompt_text = data["prompt"]

        data = []
        with open(filename, 'r', encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            for row in csv_reader:
                # print(row)
                data.append([
                    {
                        "role": "system",
                        "content": prompt_text
                    },
                    {
                        "role": "user",
                        "content": f"'id': '{row[0]}', 'term': '{row[1]}'\nYour prediction: 'types': '{row[2]}'"
                    }
                ])
        
        print(f"Processing {dataset_name} with {MODEL_NAME}...")

        results = []
        for item in tqdm(data, desc=f"Processing {dataset_name}"):
            # Process each item here
            result = client.chat.completions.create(
                model=MODEL_NAME,
                response_model=TermTypingWithReason,
                messages=item,
            )
            results.append(result)

        result_filename = RESULT_DIR.joinpath(MODEL_NAME).joinpath(f"{dataset_name.lower()}_results.json")
        os.makedirs(result_filename.parent, exist_ok=True)
        formatted_results = [{"id": result.id, "types": result.types, "reason": result.reason} for result in results]
        with open(result_filename, 'w') as f:
            json.dump(formatted_results, f, indent=2)

if __name__ == "__main__":
    main()