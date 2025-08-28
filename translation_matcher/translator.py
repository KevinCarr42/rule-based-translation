import json
import re
from typing import Dict, List, Tuple, Any

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    NLP_AVAILABLE = True
except (ImportError, OSError):
    print("Warning: spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")
    NLP_AVAILABLE = False


def load_translations(json_file="all_translations.json"):
    """Load all translation data from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_search_patterns(translations_data):
    """Create search patterns organized by category for efficient matching"""
    patterns = {}

    for category, translations in translations_data['translations'].items():
        # Sort by length (longest first) to match longer phrases first
        sorted_terms = sorted(translations.keys(), key=len, reverse=True)
        patterns[category] = sorted_terms

    return patterns

def preserve_capitalization(original_text: str, replacement_text: str) -> str:
    """Preserve the capitalization pattern from original text in replacement"""
    if not original_text or not replacement_text:
        return replacement_text
    
    if original_text.isupper():
        return replacement_text.upper()
    elif original_text.islower():
        return replacement_text.lower()
    elif original_text[0].isupper():
        return replacement_text.capitalize()
    else:
        return replacement_text

def detect_places_with_nlp(text: str) -> List[Tuple[int, int, str]]:
    """Use NLP to detect place names in text. Returns list of (start, end, text) tuples"""
    places = []
    
    if not NLP_AVAILABLE:
        return places
    
    doc = nlp(text)
    for ent in doc.ents:
        # Detect geographical entities
        if ent.label_ in ['GPE', 'LOC', 'FAC']:  # GPE=Geopolitical, LOC=Location, FAC=Facility
            places.append((ent.start_char, ent.end_char, ent.text))
    
    return places


def preprocess_for_translation(text: str, translations_file="all_translations.json") -> Tuple[str, Dict[str, Any]]:
    """
    Preprocess text for translation by replacing translatable terms with tokens.
    
    Args:
        text (str): Input text to preprocess
        translations_file (str): Path to translations JSON file
        
    Returns:
        Tuple[str, Dict]: (processed_text, token_mapping)
            - processed_text: Text with terms replaced by tokens
            - token_mapping: Dictionary mapping tokens to original terms and translations
    """
    translations_data = load_translations(translations_file)
    patterns = create_search_patterns(translations_data)

    processed_text = text
    token_mapping = {}
    token_counters = {
        'technical_terms': 0,
        'species_names': 0,
        'place_names': 0,
        'acronyms_abbreviations': 0,
        'nlp_places': 0  # For NLP-detected places
    }

    # Step 1: First detect NLP places and tokenize them
    nlp_places = detect_places_with_nlp(processed_text)
    for start, end, place_text in reversed(nlp_places):  # Reverse to maintain indices
        token_counters['nlp_places'] += 1
        token = f"__PLACE_{token_counters['nlp_places']:03d}__"
        
        # Check if this place has a known translation
        place_translation = None
        for place_key, translation in translations_data['translations']['place_names'].items():
            if place_key.lower() == place_text.lower():
                place_translation = translation
                break
        
        # Store mapping
        token_mapping[token] = {
            'original_text': place_text,
            'category': 'nlp_places',
            'translation': place_translation,
            'should_translate': place_translation is not None
        }
        
        # Replace in text
        processed_text = processed_text[:start] + token + processed_text[end:]

    # Step 2: Process dictionary-based terms (excluding place_names since NLP handles those)
    for category, terms in patterns.items():
        if category == 'place_names':
            continue  # Skip - handled by NLP above
            
        category_short = category.split('_')[0].upper()

        for term in terms:
            # Case-insensitive search for the term as whole words/phrases
            # Use word boundaries for single words, but allow phrase matching
            if ' ' in term:
                # Multi-word phrase - match exactly
                pattern = re.compile(re.escape(term), re.IGNORECASE)
            else:
                # Single word - use word boundaries to avoid partial matches
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)

            matches = list(pattern.finditer(processed_text))

            for match in reversed(matches):  # Reverse to maintain indices
                original_text = match.group()
                start, end = match.span()

                # Create token
                token_counters[category] += 1
                token = f"__{category_short}_{token_counters[category]:03d}__"

                # Get the correct translation key (use original case from translations)
                translation_key = None
                for original_key in translations_data['translations'][category].keys():
                    if original_key.lower() == term.lower():
                        translation_key = original_key
                        break

                # Store mapping
                token_mapping[token] = {
                    'original_text': original_text,
                    'category': category,
                    'translation': translations_data['translations'][category].get(translation_key, None) if translation_key else None,
                    'should_translate': True
                }

                # Replace in text
                processed_text = processed_text[:start] + token + processed_text[end:]

    return processed_text, token_mapping


def postprocess_translation(translated_text: str, token_mapping: Dict[str, Any]) -> str:
    """
    Postprocess translated text by replacing tokens with appropriate translations.
    
    Args:
        translated_text (str): Text output from translation model
        token_mapping (Dict): Token mapping from preprocess_for_translation
        
    Returns:
        str: Final text with tokens replaced by proper translations or original terms
    """
    result_text = translated_text

    # Process tokens - extract number from token format __CATEGORY_NUM__
    def extract_token_number(token):
        try:
            parts = token.split('_')
            # Find the numeric part (should be the last part before the final __)
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
            return 0
        except:
            return 0

    # Sort tokens by number in reverse order
    tokens = sorted(token_mapping.keys(), key=extract_token_number, reverse=True)

    for token in tokens:
        if token in result_text:
            mapping = token_mapping[token]

            if mapping['should_translate'] and mapping['translation'] and mapping['translation'] != 'None':
                # Use the translation with preserved capitalization
                replacement = preserve_capitalization(mapping['original_text'], mapping['translation'])
            else:
                # Use original text (no translation available or shouldn't translate)
                replacement = mapping['original_text']

            result_text = result_text.replace(token, replacement)

    return result_text


def get_translation_statistics(token_mapping: Dict[str, Any]) -> Dict[str, int]:
    """Get statistics about what was found and processed"""
    stats = {}
    for token, mapping in token_mapping.items():
        category = mapping['category']
        if category not in stats:
            stats[category] = 0
        stats[category] += 1

    return stats
