#!/usr/bin/env python3
"""
Script to join results from results_best with test datasets and create CSV files.
"""

import json
import pandas as pd
import os
from pathlib import Path


def load_json(file_path):
    """Load JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_dataset_test_files():
    """Get mapping of dataset names to their test data files."""
    datasets_dir = Path("datasets")
    dataset_mapping = {}
    
    for dataset_folder in datasets_dir.iterdir():
        if dataset_folder.is_dir():
            test_dir = dataset_folder / "test"
            if test_dir.exists():
                # Find the test data file
                for test_file in test_dir.glob("*_test_data.json"):
                    dataset_name = dataset_folder.name.lower()
                    dataset_mapping[dataset_name] = test_file
                    print(f"Found test data for {dataset_name}: {test_file}")
    
    return dataset_mapping


def get_result_files():
    """Get all result files from results_best folder."""
    results_dir = Path("results_best")
    result_files = []
    
    for model_folder in results_dir.iterdir():
        if model_folder.is_dir():
            for result_file in model_folder.glob("*.json"):
                dataset_name = result_file.stem  # filename without extension
                model_name = model_folder.name
                result_files.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'file_path': result_file
                })
                print(f"Found result file: {model_name}/{dataset_name}.json")
    
    return result_files


def join_data(test_data, result_data):
    """Join test data with result data on 'id' field."""
    # Create dictionaries for faster lookup
    test_dict = {item['id']: item for item in test_data}
    result_dict = {item['id']: item for item in result_data}
    
    joined_data = []
    
    # Join on common IDs
    for test_id, test_item in test_dict.items():
        if test_id in result_dict:
            result_item = result_dict[test_id]
            
            # Create joined record
            types_data = result_item.get('types', [])
            
            # Convert types list to string for CSV
            if isinstance(types_data, list):
                types_str = '; '.join(types_data)
            else:
                types_str = str(types_data)
            
            joined_record = {
                'id': test_id,
                'term': test_item.get('term', ''),
                'types': types_str
            }
            
            joined_data.append(joined_record)
    
    return joined_data


def create_csv_output(joined_data, output_path):
    """Create CSV file from joined data."""
    if not joined_data:
        print(f"No data to write to {output_path}")
        return
    
    # Create DataFrame
    df = pd.DataFrame(joined_data)
    
    # Keep only the needed columns
    columns_order = ['id', 'term', 'types']
    df = df[columns_order]
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved {len(joined_data)} records to {output_path}")


def main():
    """Main function to process all datasets and models."""
    # Get dataset test files
    dataset_mapping = get_dataset_test_files()
    
    # Get result files
    result_files = get_result_files()
    
    # Create output directory
    output_dir = Path("need_reason_data")
    output_dir.mkdir(exist_ok=True)
    
    # Process each result file
    for result_info in result_files:
        model_name = result_info['model']
        dataset_name = result_info['dataset']
        result_file_path = result_info['file_path']
        
        print(f"\nProcessing {model_name}/{dataset_name}...")
        
        # Find corresponding test data file
        test_file_path = None
        for ds_name, test_path in dataset_mapping.items():
            if ds_name == dataset_name:
                test_file_path = test_path
                break
        
        if not test_file_path:
            print(f"Warning: No test data found for dataset '{dataset_name}'")
            continue
        
        # Load data
        test_data = load_json(test_file_path)
        result_data = load_json(result_file_path)
        
        if test_data is None or result_data is None:
            print(f"Skipping {model_name}/{dataset_name} due to loading errors")
            continue
        
        # Join data
        joined_data = join_data(test_data, result_data)
        
        # Create output file path in model folder structure
        model_output_dir = output_dir / model_name
        output_filename = f"{dataset_name}.csv"
        output_path = model_output_dir / output_filename
        
        # Save to CSV
        create_csv_output(joined_data, output_path)
        
        # Print summary
        print(f"  Test data records: {len(test_data)}")
        print(f"  Result data records: {len(result_data)}")
        print(f"  Joined records: {len(joined_data)}")
        
        if len(joined_data) != len(test_data):
            print(f"  Warning: {len(test_data) - len(joined_data)} test records not found in results")


if __name__ == "__main__":
    # Change to script directory to ensure relative paths work
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("Starting data joining process...")
    main()
    print("\nData joining process completed!")
