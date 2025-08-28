import pandas as pd

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