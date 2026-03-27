import sys
import argparse
from pathlib import Path

# Add the 'src' directory to sys.path to allow running as script directly
src_path = str(Path(__file__).resolve().parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not installed. Install it with: pip install PyMuPDF")
    sys.exit(1)

from docx import Document
from docx.shared import Pt, Cm

# Import The existing processing pipeline for scanned PDFs
try:
    from pdf_processing.processs_pdf_to_docs import process_pdf_to_docx
except ImportError:
    print("Warning: Could not import process_pdf_to_docx from pdf_processing.processs_pdf_to_docs")
    process_pdf_to_docx = None

def analyze_pdf(pdf_path: Path):
    """Check if PDF is scanned or text-based based on check_pdf_type.py logic"""
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
    return text_ratio, total_pages, text_pages

def convert_regular_pdf(pdf_path: Path, output_docx: Path):
    """Convert a TEXT-BASED PDF to DOCX using PyMuPDF while attempting to preserve paragraph reading order"""
    print(f"\n[PyMuPDF] Converting text-based PDF to DOCX...")
    doc = fitz.open(pdf_path)
    docx = Document()
    
    # Set page margins to standard A4 defaults
    sections = docx.sections
    for section in sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)
        
    for page_num in range(len(doc)):
        page = doc[page_num]
        print(f"  Processing page {page_num + 1}/{len(doc)}...")
        
        # Extract text blocks
        blocks = page.get_text("blocks")
        
        # Extracted blocks format: (x0, y0, x1, y1, text, block_type, block_no)
        # block_type == 0 indicates text
        
        # Sort blocks by vertical position then horizontal position to get natural reading order
        blocks.sort(key=lambda b: (b[1], b[0]))
        
        for b in blocks:
            # We only care about text blocks (type 0)
            if len(b) >= 7 and b[6] == 0:
                text = b[4].strip()
                if text:
                    # Add simple paragraph
                    docx.add_paragraph(text)
        
        # Add a page break between pages
        if page_num < len(doc) - 1:
            docx.add_page_break()
            
    docx.save(output_docx)
    print(f"✓ Saved regular PDF conversion to {output_docx}")

def main():
    parser = argparse.ArgumentParser(description="Auto-detect PDF type and convert to DOCX")
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument("--output", "-o", help="Path to the output DOCX file (optional)", default=None)
    parser.add_argument("--enable_ocr", action="store_true", help="Enable OCR for scanned/hybrid PDFs processing pipeline")
    
    # Process pipeline optional args
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF to Image conversion (default: 300)")
    parser.add_argument("--imgsz", type=int, default=1024, help="YOLO image size (default: 1024)")
    parser.add_argument("--conf", type=float, default=0.1, help="YOLO confidence threshold (default: 0.1)")
    parser.add_argument("--load_4bit", action="store_true", help="Load OCR model in 4-bit config")
    parser.add_argument("--load_8bit", action="store_true", help="Load OCR model in 8-bit config")
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_path).resolve()
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
        
    if args.output is None:
        output_docx = pdf_path.parent / f"{pdf_path.stem}_output.docx"
    else:
        output_docx = Path(args.output).resolve()

    print("=" * 60)
    print(f"PDF Auto-Processing Workflow: {pdf_path.name}")
    print("=" * 60)
    
    # 1. Analyze the PDF
    text_ratio, total_pages, text_pages = analyze_pdf(pdf_path)
    
    print(f"Total pages: {total_pages}")
    print(f"Pages with text: {text_pages}")
    print(f"Text ratio: {text_ratio:.1%}")
    
    # 2. Decide workflow route
    if text_ratio > 0.8:
        print("\n✓ Classification: TEXT-BASED PDF")
        print("✓ Action: Using PyMuPDF fast text extraction")
        # Route A: PyMuPDF for regular text-based PDFs
        convert_regular_pdf(pdf_path, output_docx)
        
    else:
        if text_ratio > 0.3:
            print("\n⚠️  Classification: HYBRID PDF (Some text, some scanned/image-based)")
        else:
            print("\n✗ Classification: SCANNED/IMAGE-BASED PDF")
            
        print("✓ Action: Using processs_pdf_to_docs pipeline for advanced layout reconstruction")
        # Route B: Complex pipeline for scanned PDFs
        
        if process_pdf_to_docx is None:
            print("Error: process_pdf_to_docx function is not available. Please ensure processs_pdf_to_docs.py is in the exact same directory.")
            sys.exit(1)
            
        process_pdf_to_docx(
            pdf_path=str(pdf_path), 
            output_docx=str(output_docx),
            imgsz=args.imgsz,
            conf=args.conf,
            dpi=args.dpi,
            enable_ocr=args.enable_ocr,
            load_4bit=args.load_4bit,
            load_8bit=args.load_8bit
        )

if __name__ == "__main__":
    main()
