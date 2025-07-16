import json
import os
import re

def find_missing_data(jsonl_file, results_file=None, reference_file=None):
    # Initialize the missing data dictionary
    missing_data = {
        "missing_in_results": [],
        "summary": {}
    }

    # Load the JSONL file
    jsonl_data = []
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                jsonl_data.append(json.loads(line.strip()))
        print(f"Loaded {len(jsonl_data)} entries from the JSONL file")
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        return {"error": f"Failed to load JSONL file: {str(e)}"}

    # Load the results file if provided
    results_data = None
    if results_file:
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            print(f"Loaded results file with {len(results_data) if isinstance(results_data, list) else 'a'} entries")
        except Exception as e:
            print(f"Error loading results file: {e}")
            return {"error": f"Failed to load results file: {str(e)}"}

    # Load the reference file if provided
    reference_data = None
    if reference_file:
        try:
            with open(reference_file, 'r', encoding='utf-8') as f:
                reference_data = json.load(f)
            print(f"Loaded reference file with {len(reference_data) if isinstance(reference_data, list) else 'a'} entries")
        except Exception as e:
            print(f"Error loading reference file: {e}")
            return {"error": f"Failed to load reference file: {str(e)}"}

    # Extract IDs from reference data
    reference_ids = set()
    if reference_data and isinstance(reference_data, list):
        for item in reference_data:
            if 'id' in item:
                reference_ids.add(item['id'])
        print(f"Found {len(reference_ids)} unique IDs in reference data")
    
    # Extract IDs from results data
    result_ids = set()
    if results_data:
        if isinstance(results_data, list):
            for item in results_data:
                if 'id' in item:
                    result_ids.add(item['id'])
        elif isinstance(results_data, dict):
            # Handle the case where results might be stored as values in a dictionary
            for key, item in results_data.items():
                if isinstance(item, dict) and 'id' in item:
                    result_ids.add(item['id'])
        print(f"Found {len(result_ids)} unique IDs in results data")
    
    # Find missing IDs in results
    missing_ids = reference_ids - result_ids
    print(f"Found {len(missing_ids)} missing IDs")
    
    # Find corresponding prompts in JSONL data for missing IDs
    for missing_id in missing_ids:
        for item in jsonl_data:
            # Search for the ID in the content of user messages
            for message in item:
                if message['role'] == 'user' and 'content' in message:
                    content = message['content']
                    # Look for ID pattern in content
                    id_match = re.search(r'id:\s*(' + re.escape(missing_id) + r')\b', content)
                    if id_match:
                        missing_data["missing_in_results"].append({
                            "id": missing_id,
                            "prompt": item
                        })
                        break
    
    # Add summary information
    missing_data["summary"] = {
        "total_reference_ids": len(reference_ids),
        "total_result_ids": len(result_ids),
        "total_missing_ids": len(missing_ids),
        "missing_ids": list(missing_ids),
        "missing_prompts_found": len(missing_data["missing_in_results"])
    }
    
    return missing_data

if __name__ == "__main__":
    # Paths to your files
    jsonl_file = "processed_datasets_judge/sweet/gemini-2.5-pro/sweet_gpt-4o_deepseek-chat_claude-sonnet-4-20250514_test.jsonl"
    results_file = "results_judge/gemini-2.5-pro/sweet_gpt-4o_deepseek-chat_claude-sonnet-4-20250514_result.json"
    reference_file = "datasets/SWEET/test/sweet_term_typing_test_data.json"
    
    # Run the function
    missing_data = find_missing_data(jsonl_file, results_file, reference_file)
    
    # Save the full report
    output_file = "missing_data_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(missing_data, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Also save just the missing prompts in JSONL format
    missing_prompts_file = "missing_prompts.jsonl"
    with open(missing_prompts_file, 'w', encoding='utf-8') as f:
        for missing_item in missing_data["missing_in_results"]:
            # Write each prompt as a separate line in JSONL format
            f.write(json.dumps(missing_item["prompt"]) + "\n")
    
    print(f"Missing prompts saved to {missing_prompts_file}")