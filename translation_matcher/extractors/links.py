import openpyxl

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