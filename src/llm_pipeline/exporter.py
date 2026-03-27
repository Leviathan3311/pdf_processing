"""
Exporter - Step 5: Save revised documents.

Handles saving revised .docx files and optional PDF export.
"""
from pathlib import Path
from typing import Optional


def export_docx(docx_path: str, output_dir: str = None, output_filename: str = None) -> str:
    """
    Copy/save a .docx file to the output directory.
    
    Args:
        docx_path: path to the source .docx
        output_dir: target directory
        output_filename: custom filename
    
    Returns:
        Path to the saved file
    """
    import shutil
    
    source = Path(docx_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")
    
    if output_dir is None:
        output_dir = source.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_filename is None:
        output_filename = source.name
    
    dest = output_dir / output_filename
    
    if source != dest:
        shutil.copy2(source, dest)
    
    return str(dest)


def export_pdf(docx_path: str, output_dir: str = None, output_filename: str = None) -> Optional[str]:
    """
    Convert .docx to PDF.
    
    Tries docx2pdf first (Windows, requires MS Word installed),
    then falls back to libreoffice CLI.
    
    Returns:
        Path to the PDF file, or None if conversion failed.
    """
    source = Path(docx_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")
    
    if output_dir is None:
        output_dir = source.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_filename is None:
        output_filename = f"{source.stem}.pdf"
    
    pdf_path = output_dir / output_filename
    
    # Try docx2pdf (Windows with MS Word)
    try:
        from docx2pdf import convert
        convert(str(source), str(pdf_path))
        print(f"[Exporter] ✓ Exported PDF: {pdf_path}")
        return str(pdf_path)
    except ImportError:
        pass
    except Exception as e:
        print(f"[Exporter] ⚠️ docx2pdf failed: {e}")
    
    # Fallback: libreoffice CLI
    try:
        import subprocess
        result = subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(output_dir), str(source)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            print(f"[Exporter] ✓ Exported PDF via LibreOffice: {pdf_path}")
            return str(pdf_path)
        else:
            print(f"[Exporter] ⚠️ LibreOffice failed: {result.stderr}")
    except FileNotFoundError:
        print("[Exporter] ⚠️ Neither docx2pdf nor LibreOffice available for PDF export.")
    except Exception as e:
        print(f"[Exporter] ⚠️ PDF export failed: {e}")
    
    return None
