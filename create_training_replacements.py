import json
import re
from scipy.stats import pareto


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_translations(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['translations']


def build_term_index(translations):
    french_to_info = {}
    english_to_info = {}
    
    for category, terms in translations.items():
        for french_term, english_term in terms.items():
            french_to_info[french_term] = (category, french_term, english_term)
            english_to_info[english_term] = (category, french_term, english_term)
    
    return french_to_info, english_to_info


def find_translation_matches(source, target, source_lang, french_index, english_index):
    matches = []
    
    if source_lang == 'en':
        for english_term in english_index:
            if english_term in source:
                category, french_term, _ = english_index[english_term]
                if french_term in target:
                    matches.append((category, french_term, english_term))
    else:
        for french_term in french_index:
            if french_term in source:
                category, _, english_term = french_index[french_term]
                if english_term in target:
                    matches.append((category, french_term, english_term))
    
    return matches


def replace_whole_word(text, word, replacement):
    pattern = r'(?<!\S)' + re.escape(word) + r'(?=\s|[.,;:!?]|$)'
    return re.sub(pattern, replacement, text)


def create_replacement_token(category, counter):
    return f"{category.upper()}{counter:04d}"


def choose_random_int():
    max_n = 999
    n = int(pareto(b=1.16, scale=1).rvs())
    if n <= max_n:
        return n
    return choose_random_int()


def process_training_data():
    print("Loading data...")
    training_data = load_jsonl('../Data/training_data.jsonl')
    translations = load_translations('../Data/preferential_translations.json')
    
    print("Building indexes...")
    french_index, english_index = build_term_index(translations)
    
    print(f"Processing {len(training_data)} entries...")
    results = []
    
    for i, entry in enumerate(training_data):
        if i % 5000 == 0:
            print(f"Processed {i}/{len(training_data)}")
        
        source = entry['source']
        target = entry['target']
        source_lang = entry['source_lang']
        
        matches = find_translation_matches(source, target, source_lang, french_index, english_index)
        
        if matches:
            new_source = source
            new_target = target
            local_counters = {}
            
            for category, french_term, english_term in matches:
                if category not in local_counters:
                    local_counters[category] = choose_random_int()
                else:
                    local_counters[category] += 1
                
                replacement_token = create_replacement_token(category, local_counters[category])
                
                new_source = replace_whole_word(new_source, french_term, replacement_token)
                new_source = replace_whole_word(new_source, english_term, replacement_token)
                new_target = replace_whole_word(new_target, french_term, replacement_token)
                new_target = replace_whole_word(new_target, english_term, replacement_token)
            
            results.append({
                'source': new_source,
                'target': new_target,
                'source_lang': source_lang
            })
    
    print(f"Writing {len(results)} results to training_replacements.jsonl...")
    with open('../Data/training_replacements.jsonl', 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Completed! Found {len(results)} entries with valid translation matches")


if __name__ == "__main__":
    process_training_data()
