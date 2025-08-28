from translation_matcher import generate_all_translations

if __name__ == "__main__":
    all_data = generate_all_translations(
        spreadsheet_file="translations_spreadsheet.xlsx",
        output_file="all_translations.json"
    )

    print("\nData structure keys:", list(all_data.keys()))
    print("Translation categories:", list(all_data['translations'].keys()))
