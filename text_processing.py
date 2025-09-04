import json
import re
from typing import Dict, List, Tuple, Any, Optional
import subprocess
import sys
import spacy


def ensure_spacy_model(model_name="en_core_web_sm"):
    try:
        nlp = spacy.load(model_name)
        return nlp
    except OSError:
        print(f"Model '{model_name}' not found. Downloading now...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        print(f"Model '{model_name}' downloaded successfully!")
        return spacy.load(model_name)


nlp = ensure_spacy_model("en_core_web_sm")


class SafeTokenGenerator:
    def __init__(self):
        self.counter = 1
        self.token_map = {}
        self.reverse_map = {}
    
    def create_token(self, original_text: str) -> str:
        if original_text in self.reverse_map:
            return self.reverse_map[original_text]
        
        if self.counter > 1024:
            raise ValueError("Exceeded maximum number of tokens (1024)")
        
        token = f"<KEEP{self.counter}>"
        
        self.token_map[token] = original_text
        self.reverse_map[original_text] = token
        self.counter += 1
        
        return token
    
    def get_original(self, token: str) -> Optional[str]:
        return self.token_map.get(token, None)
    
    def replace_all_tokens(self, text: str) -> str:
        if not self.token_map:
            return text
        
        result = text
        for token, original in self.token_map.items():
            result = result.replace(token, original)
        
        return result


def load_translations(json_file: str = "all_translations.json") -> Dict:
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_search_patterns(translations_data: Dict) -> Dict[str, List[str]]:
    patterns = {}
    
    for category, translations in translations_data['translations'].items():
        sorted_terms = sorted(translations.keys(), key=len, reverse=True)
        patterns[category] = sorted_terms
    
    return patterns


def preserve_capitalization(original: str, replacement: str) -> str:
    if not original or not replacement:
        return replacement
    
    if original.isupper():
        return replacement.upper()
    elif original.islower():
        return replacement.lower()
    elif original[0].isupper() and len(original) > 1:
        if original[1:].islower():
            return replacement.capitalize()
        else:
            return replacement
    return replacement


def detect_places_with_nlp(text: str) -> List[Tuple[int, int, str]]:
    places = []
    
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC', 'FAC']:
            places.append((ent.start_char, ent.end_char, ent.text))
    
    return places


def preprocess_for_translation(
        text: str,
        translations_file: str = "all_translations.json"
) -> Tuple[str, SafeTokenGenerator, Dict[str, Tuple[str, str]]]:
    translations_data = load_translations(translations_file)
    patterns = create_search_patterns(translations_data)
    
    processed_text = text
    token_generator = SafeTokenGenerator()
    translation_map = {}
    
    tokenized_spans = []
    
    def is_overlapping(start: int, end: int) -> bool:
        for tok_start, tok_end in tokenized_spans:
            if not (end <= tok_start or start >= tok_end):
                return True
        return False
    
    nlp_places = detect_places_with_nlp(processed_text)
    nlp_places.sort(key=lambda x: x[0], reverse=True)
    
    for start, end, place_text in nlp_places:
        if is_overlapping(start, end):
            continue
        
        place_translation = None
        for place_key, translation in translations_data['translations'].get('place_names', {}).items():
            if place_key.lower() == place_text.lower():
                place_translation = translation
                break
        
        token = token_generator.create_token(place_text)
        
        if place_translation and place_translation != 'None':
            translation_map[token] = (place_text, place_translation)
        
        processed_text = processed_text[:start] + token + processed_text[end:]
        tokenized_spans.append((start, end))
    
    for category, terms in patterns.items():
        if category == 'place_names':
            continue
        
        for term in terms:
            if ' ' in term:
                pattern = re.compile(re.escape(term), re.IGNORECASE)
            else:
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            
            matches = list(pattern.finditer(processed_text))
            
            for match in reversed(matches):
                start, end = match.span()
                
                if is_overlapping(start, end):
                    continue
                
                original_text = match.group()
                
                translation_key = None
                for original_key in translations_data['translations'][category].keys():
                    if original_key.lower() == term.lower():
                        translation_key = original_key
                        break
                
                translation = (translations_data['translations'][category].get(translation_key, None)
                               if translation_key else None)
                
                token = token_generator.create_token(original_text)
                
                if translation and translation != 'None':
                    translation_with_caps = preserve_capitalization(original_text, translation)
                    translation_map[token] = (original_text, translation_with_caps)
                
                processed_text = processed_text[:start] + token + processed_text[end:]
                tokenized_spans.append((start, end))
    
    return processed_text, token_generator, translation_map


def postprocess_translation(
        translated_text: str,
        token_generator: SafeTokenGenerator,
        translation_map: Dict[str, Tuple[str, str]]
) -> str:
    result_text = translated_text
    
    for token in token_generator.token_map.keys():
        if token in result_text:
            if token in translation_map:
                original, translation = translation_map[token]
                result_text = result_text.replace(token, translation)
            else:
                original = token_generator.get_original(token)
                if original:
                    result_text = result_text.replace(token, original)
    
    return result_text


def get_translation_statistics(
        token_generator: SafeTokenGenerator,
        translation_map: Dict[str, Tuple[str, str]]
) -> Dict[str, Any]:
    return {
        "total_tokens": len(token_generator.token_map),
        "tokens_with_translations": len(translation_map),
        "tokens_without_translations": len(token_generator.token_map) - len(translation_map),
        "unique_terms": len(token_generator.reverse_map)
    }
