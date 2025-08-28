import pandas as pd
import os

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