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
OUTPUT_DIR = Path("../processed_datasets_judge")
RESULT_DIR = Path("../results_judge")
MODEL_NAME = "deepseek-chat"
MAX_CONCURRENT = 5

client = instructor.from_openai(OpenAI(api_key=api_key, base_url="https://api.deepseek.com"))

class TermTyping(BaseModel):
    id: str
    types: list[str]
    reason: str

async def process_item(item, semaphore):
    async with semaphore:
        result = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL_NAME,
            response_model=TermTyping,
            messages=item,
        )
        return result

async def process_dataset(dataset_name):
    folder_name = OUTPUT_DIR.joinpath(dataset_name.lower()).joinpath(MODEL_NAME)

    dataset_pattern = f"{dataset_name.lower()}*.jsonl"
    all_files = list(folder_name.glob(dataset_pattern))

    if not all_files:
        print(f"No files found for {dataset_name} in {folder_name}")
        return

    for filename in all_files:
        print(f"Processing file: {filename}")
        data = []
        with open(filename, encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        
        print(f"Processing {len(data)} items from {filename.name} with {MODEL_NAME}...")
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        
        tasks = [process_item(item, semaphore) for item in data]
        file_results = []
        
        for fut in tqdm_asyncio.as_completed(tasks, desc=f"Processing {filename.name}", total=len(tasks)):
            try:
                result = await fut
                file_results.append(result)
            except Exception as e:
                print(f"Error processing item: {e}")
        
        # Create result filename by replacing '_test' with '_result'
        base_name = filename.stem
        if '_test' in base_name:
            result_file_stem = base_name.replace('_test', '_result')
        else:
            result_file_stem = f"{base_name}_result"
        
        result_filename = RESULT_DIR.joinpath(MODEL_NAME).joinpath(f"{result_file_stem}.json")
        
        os.makedirs(result_filename.parent, exist_ok=True)
        formatted_results = [{"id": r.id, "types": r.types, "reason": r.reason} for r in file_results]
        
        with open(result_filename, 'w') as f:
            json.dump(formatted_results, f, indent=2)
        
        print(f"Saved results to {result_filename}")

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