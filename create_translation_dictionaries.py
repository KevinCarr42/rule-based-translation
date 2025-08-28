import pandas as pd
import openpyxl
import json


def extract_technical_terms(file_path):
    """Extract French-English translations from Technical Terms sheet"""
    df = pd.read_excel(file_path, sheet_name="Technical Terms")

    translations = {}

    for _, row in df.iterrows():
        en_term = row.get('Term (E)')
        fr_term = row.get('Term (F)')
        alt_fr = row.get('Alternate (F)')

        if pd.notna(en_term) and pd.notna(fr_term):
            # Clean up terms
            en_clean = str(en_term).strip()
            fr_clean = str(fr_term).strip()

            if en_clean and fr_clean:
                translations[fr_clean] = en_clean

                # Add alternate French term if available
                if pd.notna(alt_fr):
                    alt_clean = str(alt_fr).strip()
                    if alt_clean:
                        translations[alt_clean] = en_clean

    return translations


def extract_species_names(file_path):
    """Extract French-English translations from Species Names sheet"""
    df = pd.read_excel(file_path, sheet_name="Species Names")

    translations = {}

    for _, row in df.iterrows():
        en_species = row.get('Species Name (E)')
        fr_species = row.get('Species Name (F)')

        if pd.notna(en_species) and pd.notna(fr_species):
            en_clean = str(en_species).strip()
            fr_clean = str(fr_species).strip()

            if en_clean and fr_clean:
                translations[fr_clean] = en_clean

    return translations


def extract_acronyms_abbreviations(file_path):
    """Extract French-English translations from Acronyms & Abbreviations sheet"""
    df = pd.read_excel(file_path, sheet_name="Aconyms & Abbreviations")

    translations = {}

    for _, row in df.iterrows():
        en_acronym = row.get('Acronym/\nAbbreviation (E) ')
        fr_acronym = row.get('Acronym/\nAbbreviation (F) ')
        en_full = row.get('Full Name/\nMeaning (E)')
        fr_full = row.get('Full Name/\nMeaning (F)')

        # Map French acronym to English acronym
        if pd.notna(en_acronym) and pd.notna(fr_acronym):
            en_clean = str(en_acronym).strip()
            fr_clean = str(fr_acronym).strip()
            if en_clean and fr_clean:
                translations[fr_clean] = en_clean

        # Map French full name to English full name
        if pd.notna(en_full) and pd.notna(fr_full):
            en_full_clean = str(en_full).strip()
            fr_full_clean = str(fr_full).strip()
            if en_full_clean and fr_full_clean:
                translations[fr_full_clean] = en_full_clean

    return translations


def get_place_names_links(file_path):
    """Extract hyperlinks from Place Names sheet"""
    wb = openpyxl.load_workbook(file_path)
    ws = wb["Place Names"]

    links = []
    for row in ws.iter_rows():
        for cell in row:
            if cell.hyperlink:
                links.append({
                    'url': cell.hyperlink.target,
                    'description': cell.value
                })

    return links


def scrape_place_names(url):
    """Attempt to scrape place names from the Canadian government website"""
    print(f"Note: The Place Names sheet contains a link to: {url}")
    print("This appears to be an interactive search interface.")
    print("For comprehensive place name translations, you would need to:")
    print("1. Use the search interface manually, or")
    print("2. Find a downloadable dataset, or")
    print("3. Use an API if available")

    # Return empty dict for now - this would need manual data collection
    return {}


def create_all_dictionaries(file_path="translations_spreadsheet.xlsx"):
    """Create all translation dictionaries from the spreadsheet"""

    print("Extracting translation dictionaries...")

    # Extract from each sheet
    technical_terms = extract_technical_terms(file_path)
    species_names = extract_species_names(file_path)
    acronyms_abbreviations = extract_acronyms_abbreviations(file_path)

    # Handle place names
    place_links = get_place_names_links(file_path)
    place_names = {}

    print(f"\nPlace Names links found:")
    for link in place_links:
        print(f"- {link['description']}: {link['url']}")

    # Create comprehensive dictionary
    all_translations = {
        'technical_terms': technical_terms,
        'species_names': species_names,
        'acronyms_abbreviations': acronyms_abbreviations,
        'place_names': place_names,
        'place_names_sources': place_links
    }

    # Save to JSON
    with open('translation_dictionaries.json', 'w', encoding='utf-8') as f:
        json.dump(all_translations, f, ensure_ascii=False, indent=2)

    # Print statistics
    print(f"\nTranslation Statistics:")
    print(f"Technical Terms: {len(technical_terms)} translations")
    print(f"Species Names: {len(species_names)} translations")
    print(f"Acronyms & Abbreviations: {len(acronyms_abbreviations)} translations")
    print(f"Place Names: {len(place_names)} translations (requires manual data collection)")

    total_direct = len(technical_terms) + len(species_names) + len(acronyms_abbreviations)
    print(f"Total direct translations available: {total_direct}")

    # Save individual dictionaries as well
    for name, dictionary in all_translations.items():
        if isinstance(dictionary, dict) and dictionary:
            with open(f'{name}_translations.json', 'w', encoding='utf-8') as f:
                json.dump(dictionary, f, ensure_ascii=False, indent=2)

    return all_translations


if __name__ == "__main__":
    dictionaries = create_all_dictionaries()

    # Show some examples
    print("\nExample translations:")
    for category, translations in dictionaries.items():
        if isinstance(translations, dict) and translations:
            print(f"\n{category.replace('_', ' ').title()}:")
            # Show first 5 translations
            for i, (fr, en) in enumerate(translations.items()):
                if i >= 5:
                    print(f"  ... and {len(translations) - 5} more")
                    break
                print(f"  {fr} -> {en}")
