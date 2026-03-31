"""
FastAPI Application - Document Intelligence API

Entry point for the LLM-powered document processing system.
Start with: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add src to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from api.routes import documents, chat
from api.schemas import HealthResponse
from llm_pipeline import vector_store, llm_engine, tools


# Output directory for revised documents
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    print("=" * 60)
    print("🚀 Starting Document Intelligence API")
    print("=" * 60)
    
    # 1. Initialize ChromaDB + embedding model
    print("\n[Startup] Initializing vector store...")
    vector_store.get_embedding_function()
    vector_store.get_collection()
    
    # 2. Load LLM
    print("\n[Startup] Loading LLM...")
    llm_engine.load_model(load_4bit=False, load_8bit=False)
    
    # 3. Restore doc registry from ChromaDB (if documents were previously indexed)
    print("\n[Startup] Restoring document registry...")
    existing_docs = vector_store.list_documents()
    for doc in existing_docs:
        doc_id = doc["doc_id"]
        file_name = doc["file_name"]
        # Try to find the docx file in uploads
        uploads_dir = ROOT_DIR / "uploads"
        possible_paths = [
            uploads_dir / f"{Path(file_name).stem}_converted.docx",
            uploads_dir / file_name,
        ]
        docx_path = ""
        for p in possible_paths:
            if p.exists() and p.suffix.lower() == ".docx":
                docx_path = str(p)
                break

        
        tools.register_document(doc_id, file_name, docx_path)
    
    if existing_docs:
        print(f"[Startup] ✓ Restored {len(existing_docs)} documents from ChromaDB")
    
    print("\n" + "=" * 60)
    print("✓ API ready at http://0.0.0.0:8000")
    print("✓ Docs at http://0.0.0.0:8000/docs")
    print("=" * 60)
    
    yield
    
    # Cleanup on shutdown
    print("\n[Shutdown] Cleaning up...")
    llm_engine.cleanup()
    print("[Shutdown] ✓ Done")


app = FastAPI(
    title="Document Intelligence API",
    description=(
        "LLM-powered document processing: upload Word/PDF, ask questions, "
        "compare documents, and edit content while preserving format. "
        "Powered by Qwen3-4B + LangChain Agent."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router)
app.include_router(chat.router)

# Serve chatbot UI
@app.get("/chatbot")
async def chatbot_page():
    """Serve the chatbot UI page."""
    return FileResponse(str(ROOT_DIR / "chatbot.html"))

# Mount static files (CSS, JS, images) from project root
app.mount("/static", StaticFiles(directory=str(ROOT_DIR)), name="static")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model status."""
    return HealthResponse(
        status="ok",
        llm_loaded=llm_engine._model is not None,
        embedding_loaded=vector_store._embedding_fn is not None,
        documents_count=len(tools.get_all_doc_ids()),
    )


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a generated/revised file."""
    # Search in outputs directory
    file_path = OUTPUT_DIR / filename
    if file_path.exists():
        return FileResponse(
            str(file_path),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=filename,
        )
    
    # Also check uploads directory
    uploads_path = ROOT_DIR / "uploads" / filename
    if uploads_path.exists():
        return FileResponse(
            str(uploads_path),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=filename,
        )
    
    return JSONResponse(
        status_code=404,
        content={"detail": f"File not found: {filename}"},
    )
