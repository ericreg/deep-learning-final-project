import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF file and extracts text from every page.
    """
    page_content = []
    
    try:
        # Open the PDF document
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                # Load the individual page
                page = doc.load_page(page_num)

                page_content.append(page.get_text())
        
        return page_content

    except Exception as e:
        return f"An error occurred: {e}"

# Example Usage:
pages = extract_text_from_pdf("hp1.pdf")
for i, page in enumerate(pages):
    print(f"Page {i + 1}:\n")
    print(page + "\n" + "-"*40 + "\n")

