import argparse
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
import os
import instructor
from pydantic import BaseModel
import json
from tqdm.asyncio import tqdm as tqdm_asyncio
import asyncio

load_dotenv()
api_key = os.environ["DEEPSEEK_API_KEY"]

AVAILABLE_DATASETS = ["MatOnto", "OBI", "SWEET"]
OUTPUT_DIR = Path("processed_datasets")
RESULT_DIR = Path("results")
MODEL_NAME = "deepseek-chat"

client = instructor.from_openai(OpenAI(api_key=api_key, base_url="https://api.deepseek.com"))

class TermTyping(BaseModel):
    id: str
    types: list[str]
    reason: str

async def process_item(item):
    result = await asyncio.to_thread(
        client.chat.completions.create,
        model=MODEL_NAME,
        response_model=TermTyping,
        messages=item,
    )
    return result

async def process_dataset(dataset_name):
    filename = OUTPUT_DIR.joinpath(MODEL_NAME).joinpath(f"{dataset_name.lower()}_test.jsonl")
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))

    print(f"Processing {dataset_name} with {MODEL_NAME}...")

    tasks = [process_item(item) for item in data]
    results = []

    for fut in tqdm_asyncio.as_completed(tasks, desc=f"Processing {dataset_name}", total=len(tasks)):
        result = await fut
        results.append(result)

    result_filename = RESULT_DIR.joinpath(MODEL_NAME).joinpath(f"{dataset_name.lower()}_results.json")
    os.makedirs(result_filename.parent, exist_ok=True)
    formatted_results = [{"id": r.id, "types": r.types, "reason": r.reason} for r in results]
    with open(result_filename, 'w') as f:
        json.dump(formatted_results, f, indent=2)

async def main_async(datasets_to_process):
    for dataset_name in datasets_to_process:
        await process_dataset(dataset_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=AVAILABLE_DATASETS + ["all"],
        help="Dataset to process or 'all' to process all datasets",
    )
    args = parser.parse_args()
    datasets_to_process = AVAILABLE_DATASETS if args.dataset == "all" else [args.dataset]
    asyncio.run(main_async(datasets_to_process))

if __name__ == "__main__":
    main()