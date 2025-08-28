from .extractors.spreadsheet import extract_technical_terms, extract_species_names, extract_acronyms_abbreviations
from .extractors.place_names import extract_place_names
from .extractors.links import get_place_names_sources
from .utils import save_json
import json
from datetime import datetime

def generate_all_translations(spreadsheet_file="translations_spreadsheet.xlsx", output_file="all_translations.json"):
    """
    Generate all translation dictionaries from source data and save as nested JSON.
    
    Args:
        spreadsheet_file (str): Path to the Excel spreadsheet file
        output_file (str): Path for the output JSON file
        
    Returns:
        dict: Nested dictionary containing all translation data
    """
    print(f"Generating translation dictionaries from {spreadsheet_file}...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Extract all data
    technical_terms = extract_technical_terms(spreadsheet_file)
    species_names = extract_species_names(spreadsheet_file)
    acronyms_abbreviations = extract_acronyms_abbreviations(spreadsheet_file)
    place_names = extract_place_names()
    place_names_sources = get_place_names_sources(spreadsheet_file)
    
    # Create nested structure
    all_translations = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'source_spreadsheet': spreadsheet_file,
            'total_categories': 4
        },
        'translations': {
            'technical_terms': technical_terms,
            'species_names': species_names,
            'acronyms_abbreviations': acronyms_abbreviations,
            'place_names': place_names
        },
        'sources': {
            'place_names_links': place_names_sources
        },
        'statistics': {
            'technical_terms_count': len(technical_terms),
            'species_names_count': len(species_names),
            'acronyms_abbreviations_count': len(acronyms_abbreviations),
            'place_names_count': len(place_names),
            'total_translations': len(technical_terms) + len(species_names) + len(acronyms_abbreviations) + len(place_names)
        }
    }
    
    # Save to file
    save_json(all_translations, output_file)
    
    # Print statistics
    print("\n" + "="*50)
    print("TRANSLATION STATISTICS")
    print("="*50)
    stats = all_translations['statistics']
    print(f"Technical Terms: {stats['technical_terms_count']}")
    print(f"Species Names: {stats['species_names_count']}")
    print(f"Acronyms & Abbreviations: {stats['acronyms_abbreviations_count']}")
    print(f"Place Names: {stats['place_names_count']}")
    print("-" * 50)
    print(f"Total Translations: {stats['total_translations']}")
    print(f"\nSaved to: {output_file}")
    
    return all_translations

__all__ = ['generate_all_translations']