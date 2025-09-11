import json
import random
from collections import defaultdict, Counter
import re


def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def extract_special_tokens(text):
    pattern = r'(NOMENCLATURE|TAXON|ACRONYM|SITE)\d+'
    return re.findall(pattern, text)


def get_token_contexts(text, tokens):
    contexts = []
    for token in tokens:
        pattern = rf'\b\w+\s+{re.escape(token)}\s+\w+\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        contexts.extend(matches)
    return contexts


def sample_training_data(input_file, output_file, target_samples=25000, general_ratio=0.15):
    print(f"Loading data from {input_file}...")
    data = load_jsonl(input_file)
    print(f"Loaded {len(data)} samples")
    
    # Separate samples with and without special tokens
    special_samples = []
    general_samples = []
    token_type_counts = defaultdict(list)
    
    for i, sample in enumerate(data):
        source_tokens = extract_special_tokens(sample['source'])
        target_tokens = extract_special_tokens(sample['target'])
        all_tokens = source_tokens + target_tokens
        
        if all_tokens:
            special_samples.append((i, sample, all_tokens))
            # Group by token type for balanced sampling
            for token in set(all_tokens):
                token_type = token.split('0')[0]  # Get NOMENCLATURE, TAXON, etc.
                token_type_counts[token_type].append((i, sample, all_tokens))
        else:
            general_samples.append((i, sample))
    
    print(f"Special token samples: {len(special_samples)}")
    print(f"General samples: {len(general_samples)}")
    print(f"Token type distribution: {dict((k, len(v)) for k, v in token_type_counts.items())}")
    
    # Calculate target counts
    general_target = int(target_samples * general_ratio)
    special_target = target_samples - general_target
    
    # Sample general translation pairs
    selected_general = random.sample(general_samples, min(general_target, len(general_samples)))
    print(f"Selected {len(selected_general)} general samples")
    
    # Sample special token samples with balanced token types
    selected_special = []
    if token_type_counts:
        samples_per_type = special_target // len(token_type_counts)
        remainder = special_target % len(token_type_counts)
        
        for i, (token_type, type_samples) in enumerate(token_type_counts.items()):
            type_target = samples_per_type + (1 if i < remainder else 0)
            type_selected = random.sample(type_samples, min(type_target, len(type_samples)))
            selected_special.extend(type_selected)
            print(f"Selected {len(type_selected)} {token_type} samples (target: {type_target})")
    
    # If we still need more samples, randomly select from remaining special samples
    if len(selected_special) < special_target:
        used_indices = set(item[0] for item in selected_special)
        remaining_special = [item for item in special_samples if item[0] not in used_indices]
        additional_needed = special_target - len(selected_special)
        additional_samples = random.sample(remaining_special, min(additional_needed, len(remaining_special)))
        selected_special.extend(additional_samples)
        print(f"Added {len(additional_samples)} additional special samples")
    
    # Combine and shuffle
    all_selected = [item[1] for item in selected_general] + [item[1] for item in selected_special]
    random.shuffle(all_selected)
    
    print(f"Final sample count: {len(all_selected)}")
    print(f"General samples: {len(selected_general)} ({len(selected_general) / len(all_selected) * 100:.1f}%)")
    print(f"Special samples: {len(selected_special)} ({len(selected_special) / len(all_selected) * 100:.1f}%)")
    
    # Analyze token diversity in final selection
    final_token_counts = Counter()
    context_diversity = []
    
    for sample in all_selected:
        source_tokens = extract_special_tokens(sample['source'])
        target_tokens = extract_special_tokens(sample['target'])
        all_tokens = source_tokens + target_tokens
        
        for token in all_tokens:
            token_type = token.split('0')[0]
            final_token_counts[token_type] += 1
        
        # Analyze contexts
        contexts = get_token_contexts(sample['source'], source_tokens)
        contexts.extend(get_token_contexts(sample['target'], target_tokens))
        context_diversity.extend(contexts)
    
    print(f"Final token type distribution: {dict(final_token_counts)}")
    print(f"Unique contexts found: {len(set(context_diversity))}")
    
    # Save sampled data
    save_jsonl(all_selected, output_file)
    print(f"Saved {len(all_selected)} samples to {output_file}")


if __name__ == "__main__":
    random.seed(42)  # For reproducible sampling
    sample_training_data(
        input_file="../Data/training_replacements.jsonl",
        output_file="../Data/training_replacements_sampled.jsonl",
        target_samples=25000,
        general_ratio=0.15
    )
