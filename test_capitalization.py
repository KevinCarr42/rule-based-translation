from translation_matcher import preprocess_for_translation, postprocess_translation

# Test with different capitalizations
test_cases = [
    "Le homard est important.",  # lowercase
    "Le Homard est important.",  # capitalized
    "Le HOMARD est important.",  # uppercase
]

for i, test_text in enumerate(test_cases):
    print(f"Test case {i+1}: {test_text}")
    processed_text, token_mapping = preprocess_for_translation(test_text)
    
    print(f"Processed: {processed_text}")
    
    # Show token mapping details
    for token, mapping in token_mapping.items():
        if 'homard' in mapping['original_text'].lower():
            print(f"  Token: {token}")
            print(f"  Original: '{mapping['original_text']}'")
            print(f"  Translation: '{mapping['translation']}'")
    
    # Simulate translation (keep tokens)
    simulated_output = processed_text.replace("Le ", "The ").replace("est important", "is important")
    final_text = postprocess_translation(simulated_output, token_mapping)
    
    print(f"Final: {final_text}")
    print("-" * 40)