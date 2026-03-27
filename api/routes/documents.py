"""
Document routes - Upload, list, detail, delete documents.
"""
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from api.schemas import DocumentInfo, DocumentDetail, DocumentElement, UploadResponse, DeleteResponse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from llm_pipeline import document_parser, vector_store, tools


router = APIRouter(prefix="/documents", tags=["Documents"])

# Upload directory
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a Word (.docx) or PDF (.pdf) file.
    The file will be parsed, structured, and indexed in ChromaDB.
    """
    # Validate file type
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ('.pdf', '.docx', '.doc'):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Only .pdf, .docx, .doc are supported."
        )
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / file.filename
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        # Parse and index document
        doc_structure = document_parser.ingest_file(
            str(upload_path),
            upload_dir=str(UPLOAD_DIR),
        )
        
        # Index in ChromaDB
        element_count = vector_store.add_document(doc_structure)
        
        # Register in tools registry (for agent access)
        tools.register_document(
            doc_id=doc_structure.doc_id,
            file_name=doc_structure.file_name,
            docx_path=doc_structure.docx_path,
        )
        
        return UploadResponse(
            doc_id=doc_structure.doc_id,
            file_name=doc_structure.file_name,
            element_count=element_count,
            message=f"✓ Document uploaded and indexed: {element_count} elements",
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.get("/", response_model=list[DocumentInfo])
async def list_documents():
    """List all uploaded and indexed documents."""
    docs = vector_store.list_documents()
    result = []
    for doc in docs:
        doc_info = tools.get_doc_info(doc["doc_id"])
        result.append(DocumentInfo(
            doc_id=doc["doc_id"],
            file_name=doc["file_name"],
            docx_path=doc_info.get("docx_path", "") if doc_info else "",
        ))
    return result


@router.get("/{doc_id}", response_model=DocumentDetail)
async def get_document(doc_id: str):
    """Get detailed information about a document including all elements."""
    elements = vector_store.get_full_document(doc_id)
    
    if not elements:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    
    doc_info = tools.get_doc_info(doc_id)
    file_name = doc_info.get("file_name", "unknown") if doc_info else "unknown"
    docx_path = doc_info.get("docx_path", "") if doc_info else ""
    
    return DocumentDetail(
        doc_id=doc_id,
        file_name=file_name,
        docx_path=docx_path,
        elements=[
            DocumentElement(
                id=el.get("metadata", {}).get("element_id", "?"),
                type=el.get("metadata", {}).get("element_type", "unknown"),
                content=el.get("content", ""),
                table_id=el.get("metadata", {}).get("table_id", None) or None,
                row=el.get("metadata", {}).get("row", None) if el.get("metadata", {}).get("row", -1) != -1 else None,
                col=el.get("metadata", {}).get("col", None) if el.get("metadata", {}).get("col", -1) != -1 else None,
            )
            for el in elements
        ],
    )


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """Delete a document from ChromaDB and the tools registry."""
    deleted_count = vector_store.delete_document(doc_id)
    
    # Remove from tools registry
    if doc_id in tools._doc_registry:
        del tools._doc_registry[doc_id]
    
    return DeleteResponse(
        doc_id=doc_id,
        deleted_elements=deleted_count,
        message=f"✓ Document deleted: {deleted_count} elements removed",
    )
