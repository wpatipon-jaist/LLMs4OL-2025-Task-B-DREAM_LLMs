import argparse
from dotenv import load_dotenv
from pathlib import Path
import os
import instructor
from pydantic import BaseModel
import json
from tqdm.asyncio import tqdm as tqdm_asyncio
import asyncio
from google import genai
import csv

load_dotenv()
api_key = os.environ["GEMINI_API_KEY"]

AVAILABLE_DATASETS = ["MatOnto", "OBI", "SWEET"]
OUTPUT_DIR = Path("../need_reason_data")
RESULT_DIR = Path("../result_with_reason")
MODEL_NAME = "gemini-2.5-pro"
MAX_CONCURRENT = 1

client = genai.Client(api_key=api_key)
client = instructor.from_genai(client)

class TermTyping(BaseModel):
    id: str
    types: list[str]
    reason: str

async def process_item(item, semaphore):
    # Wrap blocking call into a thread
    async with semaphore:
        result = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL_NAME,
            response_model=TermTyping,
            messages=item,
        )
        return result


async def process_dataset(dataset_name):
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

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        
    tasks = [process_item(item, semaphore) for item in data]
    
    results = []
    for f in tqdm_asyncio.as_completed(tasks, desc=f"Processing {dataset_name}", total=len(tasks)):
        result = await f
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