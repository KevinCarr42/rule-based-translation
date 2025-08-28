import pandas as pd
import sys

def examine_spreadsheet(file_path):
    try:
        # Read all sheets
        xl_file = pd.ExcelFile(file_path)
        print(f"Sheet names: {xl_file.sheet_names}")
        print()
        
        for sheet_name in xl_file.sheet_names:
            print(f"=== Sheet: {sheet_name} ===")
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print("First few rows:")
            print(df.head())
            print()
            
            # Check if it looks like translation data vs instructions
            if df.shape[1] >= 2:
                # Look for potential French/English columns
                fr_cols = [col for col in df.columns if 'fr' in str(col).lower() or 'french' in str(col).lower()]
                en_cols = [col for col in df.columns if 'en' in str(col).lower() or 'english' in str(col).lower()]
                print(f"Potential French columns: {fr_cols}")
                print(f"Potential English columns: {en_cols}")
            
            # Check for web links
            for col in df.columns:
                if df[col].dtype == 'object':
                    urls = df[col].dropna().astype(str).str.contains('http', case=False, na=False)
                    if urls.any():
                        print(f"URLs found in column '{col}':")
                        print(df[col][urls].tolist())
            
            print("-" * 50)
            
    except Exception as e:
        print(f"Error reading spreadsheet: {e}")

if __name__ == "__main__":
    examine_spreadsheet("translations_spreadsheet.xlsx")