import jsonlines
import json
import argparse
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATASETS_DIR = Path("datasets")
RESULT_DIR = Path("results")
AVAILABLE_DATASETS = ["MatOnto", "OBI", "SWEET"]
OUTPUT_DIR = Path("processed_datasets_judge")
MODELS = ["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.5-pro", "deepseek-chat"]


class DatasetProcessor:
    """Process ontology datasets and convert them to JSONL format."""

    def __init__(self, dataset_name: str, model_name: str, reasoners, output_dir: Path = OUTPUT_DIR):
        """
        Initialize the dataset processor.

        Args:
            dataset_name: Name of the dataset to process
            output_dir: Directory to save processed files
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.output_dir = output_dir.joinpath(dataset_name.lower()).joinpath(model_name)
        self.dataset_path = DATASETS_DIR / dataset_name
        self.result_path = RESULT_DIR
        self.reasoners = reasoners

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Validate dataset selection
        if dataset_name not in AVAILABLE_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. Available datasets: {AVAILABLE_DATASETS}"
            )

    def load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load and parse a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Parsed JSON content as a list of dictionaries
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise

    def save_jsonl(
        self, data: List[Dict[str, Any]], output_path: Path
    ) -> None:
        """
        Save data to JSONL format.

        Args:
            data: List of dictionaries to save
            output_path: Path where to save the JSONL file
        """
        try:
            with jsonlines.open(output_path, mode="w") as writer:
                writer.write_all(data)
            logger.info(f"Successfully saved {len(data)} records to {output_path}")
        except Exception as e:
            logger.error(f"Error saving to {output_path}: {e}")
            raise

    def prepare_dataset(self, result_data, test_data, labels, prompt, models):
        """
        Prepare dataset by converting it to a list of dictionaries.

        Args:
            data: Raw data from the JSON file

        Returns:
            List of dictionaries ready for batch processing with JSONL format
        """
        num_labels = len(labels)
        system_prompt = prompt.replace("[NUM_LABELS]", str(num_labels))
        system_prompt = system_prompt.replace("[LABELS]", "- " + ("\n- ".join(labels)))

        prepared_data = []
        for item in test_data:
            item_id = item["id"]
            
            # Skip items that don't have predictions from all models
            if any(item_id not in result_data[model] for model in models):
                logger.warning(f"Skipping item {item_id} as it's missing predictions from some models")
                continue
                
            model_predictions = []
            for model in models:
                prediction = result_data[model][item_id]
                model_predictions.append(f"{model} prediction:\nType: {prediction['types'][0]}\nReason: {prediction['reason']}")
            
            # Format the user prompt with the term from the test data
            user_content = f"id: {item['id']}\nterm: {item['term']}\n"
            user_content += "\n\n".join(model_predictions)
                
            batch_item = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
            
            
            prepared_data.append(batch_item)

        return prepared_data
    
    def get_labels(self, data):
        """
        Extract unique labels from the dataset.

        Args:
            data: Raw data from the JSON file

        Returns:
            List of unique labels
        """
        labels = set()
        for item in data:
            labels.add(item["types"][0])
        return list(labels)

    def process_dataset(self) -> Tuple[Path, Path]:
        """
        Process train and test files for the selected dataset.

        Returns:
            Tuple containing paths to the processed train and test JSONL files
        """
        # Load prompt data
        prompt_file = self.dataset_path / "prompt_judge.json"
        prompt = self.load_json_file(prompt_file)["prompt"]

        # Process train data
        train_file = self.dataset_path / "train" / "term_typing_train_data.json"
        train_data = self.load_json_file(train_file)
        labels = self.get_labels(train_data)

        test_file_name = f"{self.dataset_name.lower()}_term_typing_test_data.json"
        test_file = self.dataset_path / "test" / test_file_name
        test_data = self.load_json_file(test_file)

        # Process result data
        result_file_name = f"{self.dataset_name.lower()}_results.json"

        result_data = dict()

        for model in self.reasoners:
            result_file = self.result_path / model / result_file_name
            model_results = self.load_json_file(result_file)
            model_results_dict = dict()
            for result in model_results:
                model_results_dict[result["id"]] = {
                    "types": result["types"],
                    "reason": result["reason"]
                }
            result_data[model] = model_results_dict
        
        processed_test_data = self.prepare_dataset(result_data, test_data, labels, prompt, self.reasoners)
        
        str_reasonsers = '_'.join(self.reasoners)
        # train_output = self.output_dir / f"{self.dataset_name.lower()}_train.jsonl"
        test_output = self.output_dir / f"{self.dataset_name.lower()}_{str_reasonsers}_test.jsonl"
        # self.save_jsonl(train_data, train_output)
        self.save_jsonl(processed_test_data, test_output)

        return test_output

def parse_models(model_list):
    """Parse comma-separated model list and validate against available models"""
    if not model_list:
        raise ValueError(f"Invalid model")
    
    models = [m.strip() for m in model_list.split(',')]
    
    # Validate models
    for model in models:
        if model not in MODELS:
            valid_models = ", ".join(MODELS)
            raise ValueError(f"Invalid model: {model}. Available models are: {valid_models}")
    
    return models

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process ontology datasets and convert them to JSONL format."
    )
    parser.add_argument(
        "dataset",
        choices=AVAILABLE_DATASETS + ["all"],
        help="Dataset to process or 'all' to process all datasets",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(OUTPUT_DIR),
        help="Output directory for processed files",
    )
    parser.add_argument(
        "--judge",
        default=MODELS + ["all"]
    )
    parser.add_argument(
        "--reasoner",
        type=str,
        required=True
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    # Process selected dataset(s)
    datasets_to_process = AVAILABLE_DATASETS if args.dataset == "all" else [args.dataset]
    models_to_process = args.judge
    reasoners = parse_models(args.reasoner)

    for dataset_name in datasets_to_process:
        try:
            processor = DatasetProcessor(dataset_name, models_to_process, reasoners, output_dir)
            test_path = processor.process_dataset()
            logger.info(
                f"Processed {dataset_name}: Test data saved to {test_path}, For {models_to_process}"
            )
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}")


if __name__ == "__main__":
    main()

