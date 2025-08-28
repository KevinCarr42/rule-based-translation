from translation_matcher import preprocess_for_translation, postprocess_translation, get_translation_statistics


def test_translation_workflow():
    test_text = """
    Le homard americain est une espece importante pour l'aquaculture au Canada. 
    La biomasse des especes aquatiques dans le golfe du Saint-Laurent et a Toronto a ete etudiee 
    par le CCG et la region biogeographique de l'Atlantique. 
    Les facteurs abiotiques affectent la surveillance acoustique des populations a Quebec.
    """

    print("Original text:")
    print(test_text)
    print("\n" + "=" * 60 + "\n")

    # Step 1: Preprocess - replace terms with tokens
    print("Step 1: Preprocessing (tokenizing known terms)...")
    processed_text, token_mapping = preprocess_for_translation(test_text)

    print("Processed text (with tokens):")
    print(processed_text)

    print("\nToken mapping:")
    for token, mapping in token_mapping.items():
        print(f"  {token}: '{mapping['original_text']}' -> '{mapping['translation']}' ({mapping['category']})")

    print("\nStatistics:")
    stats = get_translation_statistics(token_mapping)
    for category, count in stats.items():
        print(f"  {category}: {count} terms")

    print("\n" + "=" * 60 + "\n")

    # Step 2: Simulate translation model output (normally this would go to external translator)
    print("Step 2: [Simulated] Translation model output...")
    # For demo, let's pretend the translation model converted the text but kept tokens
    simulated_translation = processed_text.replace("Le ", "The ").replace("est une", "is a").replace("importante pour", "important for")
    simulated_translation = simulated_translation.replace("au Canada", "in Canada").replace("La ", "The ")
    simulated_translation = simulated_translation.replace("des ", "of ").replace("dans le", "in the")
    simulated_translation = simulated_translation.replace("a été étudiée par", "has been studied by")
    simulated_translation = simulated_translation.replace("et la ", "and the ").replace("Les ", "The ")
    simulated_translation = simulated_translation.replace("affectent", "affect").replace("des populations", "of populations")

    print("Simulated translated text (still with tokens):")
    print(simulated_translation)

    print("\n" + "=" * 60 + "\n")

    # Step 3: Postprocess - replace tokens with proper translations
    print("Step 3: Postprocessing (replacing tokens with translations)...")
    final_text = postprocess_translation(simulated_translation, token_mapping)

    print("Final translated text:")
    print(final_text)

    return final_text


if __name__ == "__main__":
    test_translation_workflow()
