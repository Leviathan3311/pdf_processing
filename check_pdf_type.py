import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not installed. Install it with: pip install PyMuPDF")
    sys.exit(1)

def check_pdf_type(pdf_path: Path):
    """Check if PDF is scanned or text-based"""
    doc = fitz.open(pdf_path)
    
    total_pages = len(doc)
    text_pages = 0
    
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text().strip()
        if len(text) > 100:  # Has substantial text
            text_pages += 1
    
    doc.close()
    
    text_ratio = text_pages / total_pages if total_pages > 0 else 0
    
    print("=" * 60)
    print(f"PDF Analysis: {pdf_path.name}")
    print("=" * 60)
    print(f"Total pages: {total_pages}")
    print(f"Pages with text: {text_pages}")
    print(f"Text ratio: {text_ratio:.1%}")
    
    if text_ratio > 0.8:
        print("\n✓ This appears to be a TEXT-BASED PDF")
        print("  You can extract text directly without OCR")
    elif text_ratio > 0.3:
        print("\n⚠️  This appears to be a HYBRID PDF")
        print("  Some pages have text, some are scanned")
    else:
        print("\n✗ This appears to be a SCANNED PDF")
        print("  OCR is required to extract text")
    
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_pdf_type.py <pdf_file>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    check_pdf_type(pdf_path)
