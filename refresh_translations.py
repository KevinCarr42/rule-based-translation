#!/usr/bin/env python3
"""
Simple script to refresh all translation dictionaries from source data.
Run this script whenever the spreadsheet or CSV files are updated.
"""

import pandas as pd
import openpyxl
import json
import os
from datetime import datetime

def extract_technical_terms(file_path):
    """Extract French-English translations from Technical Terms sheet"""
    df = pd.read_excel(file_path, sheet_name="Technical Terms")
    
    translations = {}
    
    for _, row in df.iterrows():
        en_term = row.get('Term (E)')
        fr_term = row.get('Term (F)')
        alt_fr = row.get('Alternate (F)')
        
        if pd.notna(en_term) and pd.notna(fr_term):
            en_clean = str(en_term).strip()
            fr_clean = str(fr_term).strip()
            
            if en_clean and fr_clean:
                translations[fr_clean] = en_clean
                
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
        
        if pd.notna(en_acronym) and pd.notna(fr_acronym):
            en_clean = str(en_acronym).strip()
            fr_clean = str(fr_acronym).strip()
            if en_clean and fr_clean:
                translations[fr_clean] = en_clean
        
        if pd.notna(en_full) and pd.notna(fr_full):
            en_full_clean = str(en_full).strip()
            fr_full_clean = str(fr_full).strip()
            if en_full_clean and fr_full_clean:
                translations[fr_full_clean] = en_full_clean
    
    return translations

def extract_place_names():
    """Extract place names from CSV files in reference folder"""
    place_translations = {}
    
    csv_file = 'reference/vw_Place_Names_Noms_Lieux_APCA_V2_FGP.csv'
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            
            for _, row in df.iterrows():
                name_en = row.get('Name_e')
                name_fr = row.get('Nom_f') 
                
                if pd.notna(name_en) and pd.notna(name_fr):
                    name_en_clean = str(name_en).strip()
                    name_fr_clean = str(name_fr).strip()
                    
                    if name_en_clean != name_fr_clean and name_en_clean and name_fr_clean:
                        place_translations[name_fr_clean] = name_en_clean
                        
        except Exception as e:
            print(f"Warning: Could not process {csv_file}: {e}")
    else:
        print(f"Warning: {csv_file} not found")
    
    return place_translations

def get_place_names_sources(file_path):
    """Extract place names source links from spreadsheet"""
    try:
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
    except Exception as e:
        print(f"Warning: Could not extract place names sources: {e}")
        return []

def refresh_all_translations():
    """Main function to refresh all translation dictionaries"""
    
    spreadsheet_file = "translations_spreadsheet.xlsx"
    
    print(f"Refreshing translation dictionaries...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if not os.path.exists(spreadsheet_file):
        print(f"Error: {spreadsheet_file} not found!")
        return None
    
    # Extract from spreadsheet
    print("Extracting from spreadsheet...")
    technical_terms = extract_technical_terms(spreadsheet_file)
    species_names = extract_species_names(spreadsheet_file)
    acronyms_abbreviations = extract_acronyms_abbreviations(spreadsheet_file)
    place_names_sources = get_place_names_sources(spreadsheet_file)
    
    # Extract place names from CSV
    print("Extracting place names from CSV files...")
    place_names = extract_place_names()
    
    # Combine all translations
    all_translations = {
        'technical_terms': technical_terms,
        'species_names': species_names, 
        'acronyms_abbreviations': acronyms_abbreviations,
        'place_names': place_names,
        'place_names_sources': place_names_sources,
        'last_updated': datetime.now().isoformat()
    }
    
    # Save combined dictionary
    with open('translation_dictionaries.json', 'w', encoding='utf-8') as f:
        json.dump(all_translations, f, ensure_ascii=False, indent=2)
    
    # Save individual dictionaries
    for name, dictionary in all_translations.items():
        if isinstance(dictionary, dict) and dictionary and name != 'place_names_sources':
            filename = f'{name}_translations.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dictionary, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    print("\n" + "="*50)
    print("TRANSLATION STATISTICS")
    print("="*50)
    
    total_translations = 0
    for category, translations in all_translations.items():
        if isinstance(translations, dict):
            count = len(translations)
            total_translations += count
            print(f"{category.replace('_', ' ').title():.<30} {count:>6}")
    
    print("-" * 50)
    print(f"{'Total translations':.<30} {total_translations:>6}")
    
    # Show sample translations from each category
    print(f"\nSample translations (first 3 from each category):")
    for category, translations in all_translations.items():
        if isinstance(translations, dict) and translations:
            print(f"\n{category.replace('_', ' ').title()}:")
            for i, (fr, en) in enumerate(translations.items()):
                if i >= 3:
                    break
                print(f"  {fr} -> {en}")
    
    print(f"\nFiles updated:")
    print(f"  - translation_dictionaries.json")
    for name in all_translations.keys():
        if isinstance(all_translations[name], dict) and all_translations[name] and name != 'place_names_sources':
            print(f"  - {name}_translations.json")
    
    return all_translations

if __name__ == "__main__":
    refresh_all_translations()