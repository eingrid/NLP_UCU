import pdfplumber

def load_pdf_with_plumber(pdf_path, doc_id):
    """
    For each pdf we extract the list of tuples: (doc_id, page_id, text)
    using pdfplumber.
    """
    pages_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages_data.append((doc_id, i + 1, text)) # page numbers are 1-indexed
        return pages_data
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return []
    

