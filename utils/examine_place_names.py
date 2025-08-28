import pandas as pd

def examine_place_names():
    df = pd.read_excel("translations_spreadsheet.xlsx", sheet_name="Place Names")
    print("Place Names sheet content:")
    print(df)
    print("\nDetailed view:")
    for col in df.columns:
        print(f"Column: {col}")
        for idx, val in df[col].items():
            print(f"  Row {idx}: {val}")
        print()

if __name__ == "__main__":
    examine_place_names()