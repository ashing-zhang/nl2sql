'''
    python -m workflow.text2sql_dataset_generator.merge_datasets
'''
import json
from pathlib import Path

def merge_datasets(first_data_path,second_data_path,merged_data_path):
    # Load the first dataset
    with open(first_data_path, 'r', encoding='utf-8') as f:
        dataset1 = json.load(f)
    
    # Load the second dataset
    with open(second_data_path, 'r', encoding='utf-8') as f:
        dataset2 = json.load(f)
    
    # Create a set to track unique conversations
    seen_conversations = set()
    merged_data = []
    
    # Add conversations from first dataset
    for item in dataset1:
        conv_str = json.dumps(item['conversations'], sort_keys=True)
        if conv_str not in seen_conversations:
            seen_conversations.add(conv_str)
            merged_data.append(item)
    
    # Add conversations from second dataset
    for item in dataset2:
        conv_str = json.dumps(item['conversations'], sort_keys=True)
        if conv_str not in seen_conversations:
            seen_conversations.add(conv_str)
            merged_data.append(item)
    
    # Write merged data to new file
    with open(merged_data_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"Merged dataset created with {len(merged_data)} unique conversations")

if __name__ == '__main__':
    first_data_path = 'workflow/text2sql_dataset_generator/text2sql_dataset.json'
    second_data_path = 'workflow/text2sql_dataset_generator/text2sql_dataset_add.json'
    merged_data_path = 'workflow/text2sql_dataset_generator/merged_data.json'
    merge_datasets(first_data_path,second_data_path,merged_data_path)
