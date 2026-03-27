"""
Pydantic schemas for the Document Intelligence API.
"""
from pydantic import BaseModel, Field
from typing import Optional


# ─────────────────────────────────────────────────────
# Document schemas
# ─────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    """Summary info about an uploaded document."""
    doc_id: str
    file_name: str
    element_count: int = 0
    docx_path: str = ""


class DocumentElement(BaseModel):
    """A single element (paragraph or table cell)."""
    id: str
    type: str
    content: str
    table_id: Optional[str] = None
    row: Optional[int] = None
    col: Optional[int] = None


class DocumentDetail(BaseModel):
    """Full document detail with all elements."""
    doc_id: str
    file_name: str
    docx_path: str
    elements: list[DocumentElement] = []


class UploadResponse(BaseModel):
    """Response after uploading a document."""
    doc_id: str
    file_name: str
    element_count: int
    message: str


class DeleteResponse(BaseModel):
    """Response after deleting a document."""
    doc_id: str
    deleted_elements: int
    message: str


# ─────────────────────────────────────────────────────
# Chat schemas
# ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for the unified /chat endpoint."""
    message: str = Field(..., description="User message (Q&A, compare, edit, summarize...)")
    doc_ids: list[str] = Field(default=[], description="List of doc_ids to use as context. Empty = use all.")


class ChatResponse(BaseModel):
    """Response from the /chat endpoint."""
    response: str = Field(..., description="LLM response text")
    files: list[str] = Field(default=[], description="List of generated filenames (for edit operations)")


# ─────────────────────────────────────────────────────
# Health / Status
# ─────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    llm_loaded: bool
    embedding_loaded: bool
    documents_count: int
