import json

def save_json(data, file_path):
    """Save data to JSON file with UTF-8 encoding"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)