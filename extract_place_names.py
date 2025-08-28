import pandas as pd
import json

def extract_place_names_from_csv():
    """Extract French-English place name translations from existing CSV files"""
    
    place_translations = {}
    
    # Process the APCA place names file
    try:
        df_apca = pd.read_csv('reference/vw_Place_Names_Noms_Lieux_APCA_V2_FGP.csv')
        print(f"Processing APCA file with {len(df_apca)} records...")
        
        for _, row in df_apca.iterrows():
            name_en = row.get('Name_e')
            name_fr = row.get('Nom_f') 
            
            if pd.notna(name_en) and pd.notna(name_fr):
                name_en_clean = str(name_en).strip()
                name_fr_clean = str(name_fr).strip()
                
                # Only add if they're different (actual translations)
                if name_en_clean != name_fr_clean and name_en_clean and name_fr_clean:
                    place_translations[name_fr_clean] = name_en_clean
        
        print(f"Found {len(place_translations)} place name translations from APCA file")
                    
    except Exception as e:
        print(f"Error processing APCA file: {e}")
    
    # Process the CGN Canada file (has some French names in language field)
    try:
        df_cgn = pd.read_csv('reference/cgn_canada_csv_eng.csv')
        print(f"Processing CGN file with {len(df_cgn)} records...")
        
        french_names = 0
        for _, row in df_cgn.iterrows():
            language = row.get('Language', '')
            if 'french' in str(language).lower() or 'français' in str(language).lower():
                geo_name = row.get('Geographical Name')
                if pd.notna(geo_name):
                    french_names += 1
                    
        print(f"Found {french_names} French language entries in CGN file")
        # Note: This file would need more complex processing to extract actual translations
        
    except Exception as e:
        print(f"Error processing CGN file: {e}")
    
    return place_translations

def update_translation_dictionaries():
    """Update the main translation dictionaries with place names"""
    
    # Extract place names
    place_translations = extract_place_names_from_csv()
    
    # Load existing dictionaries
    try:
        with open('translation_dictionaries.json', 'r', encoding='utf-8') as f:
            all_translations = json.load(f)
    except FileNotFoundError:
        print("No existing translation_dictionaries.json found, creating new one")
        all_translations = {
            'technical_terms': {},
            'species_names': {},
            'acronyms_abbreviations': {},
            'place_names': {},
            'place_names_sources': []
        }
    
    # Update place names
    all_translations['place_names'] = place_translations
    
    # Save updated dictionaries
    with open('translation_dictionaries.json', 'w', encoding='utf-8') as f:
        json.dump(all_translations, f, ensure_ascii=False, indent=2)
    
    # Save place names separately
    with open('place_names_translations.json', 'w', encoding='utf-8') as f:
        json.dump(place_translations, f, ensure_ascii=False, indent=2)
    
    # Print updated statistics
    print(f"\nUpdated Translation Statistics:")
    total_translations = 0
    for category, translations in all_translations.items():
        if isinstance(translations, dict):
            count = len(translations)
            total_translations += count
            print(f"{category.replace('_', ' ').title()}: {count} translations")
    
    print(f"Total translations available: {total_translations}")
    
    # Show some place name examples
    if place_translations:
        print(f"\nPlace Name Translation Examples:")
        for i, (fr, en) in enumerate(place_translations.items()):
            if i >= 10:  # Show first 10
                print(f"  ... and {len(place_translations) - 10} more")
                break
            print(f"  {fr} -> {en}")
    
    return all_translations

if __name__ == "__main__":
    updated_dictionaries = update_translation_dictionaries()