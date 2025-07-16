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
AVAILABLE_DATASETS = ["MatOnto", "OBI", "SWEET"]
OUTPUT_DIR = Path("processed_datasets")
MODELS = ["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.5-pro", "deepseek-chat"]


class DatasetProcessor:
    """Process ontology datasets and convert them to JSONL format."""

    def __init__(self, dataset_name: str, model_name: str, output_dir: Path = OUTPUT_DIR):
        """
        Initialize the dataset processor.

        Args:
            dataset_name: Name of the dataset to process
            output_dir: Directory to save processed files
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.output_dir = output_dir.joinpath(model_name)
        self.dataset_path = DATASETS_DIR / dataset_name

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

    def prepare_dataset(self, train_data, test_data, labels, prompt):
        """
        Prepare dataset by converting it to a list of dictionaries.

        Args:
            data: Raw data from the JSON file

        Returns:
            List of dictionaries ready for batch processing with JSONL format
        """
        num_labels = len(labels)
        first_five_examples = train_data[:5]
        system_prompt = prompt
        system_prompt = prompt.replace("[NUM_LABELS]", str(num_labels))
        first_five_examples_str = ""
        for example in first_five_examples:
            first_five_examples_str += str(example) + "\n"

        system_prompt = system_prompt.replace("[FIRST_FIVE_DATASET]", first_five_examples_str)

        system_prompt = system_prompt.replace("[LABELS]", "- " + ("\n- ".join(labels)))

        prepared_data = []
        for item in test_data:
            # Assuming each item is a dictionary with relevant fields
            batch_item = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"{item}"
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
        prompt_file = self.dataset_path / "prompt.json"
        prompt = self.load_json_file(prompt_file)["prompt"]

        # Process train data
        train_file = self.dataset_path / "train" / "term_typing_train_data.json"
        train_data = self.load_json_file(train_file)
        labels = self.get_labels(train_data)

        # Process test data
        test_file_name = f"{self.dataset_name.lower()}_term_typing_test_data.json"
        test_file = self.dataset_path / "test" / test_file_name
        test_data = self.load_json_file(test_file)

        processed_test_data = self.prepare_dataset(train_data, test_data, labels, prompt)

        # train_output = self.output_dir / f"{self.dataset_name.lower()}_train.jsonl"
        test_output = self.output_dir / f"{self.dataset_name.lower()}_test.jsonl"
        # self.save_jsonl(train_data, train_output)
        self.save_jsonl(processed_test_data, test_output)

        return test_output


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
        "--model",
        default=MODELS + ["all"]
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    # Process selected dataset(s)
    datasets_to_process = AVAILABLE_DATASETS if args.dataset == "all" else [args.dataset]
    models_to_process = MODELS if args.model == "all" else [args.model]

    for dataset_name in datasets_to_process:
        for model_name in models_to_process:
            try:
                processor = DatasetProcessor(dataset_name, model_name, output_dir)
                test_path = processor.process_dataset()
                logger.info(
                    f"Processed {dataset_name}: Test data saved to {test_path}, For {model_name}"
                )
            except Exception as e:
                logger.error(f"Failed to process {dataset_name}: {e}")


if __name__ == "__main__":
    main()

