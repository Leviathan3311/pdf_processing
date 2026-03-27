"""
Vector Store - ChromaDB management with BGE-M3 embeddings.

Handles storing, searching, and managing document elements in ChromaDB.
Uses BAAI/bge-m3 for multilingual embeddings (Vietnamese + English).
"""
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import Optional

from llm_pipeline.document_parser import DocumentStructure, DocumentElement


# Singleton instances
_chroma_client = None
_embedding_fn = None


def get_embedding_function(model_name: str = "BAAI/bge-m3"):
    """Get or create the embedding function (singleton)."""
    global _embedding_fn
    if _embedding_fn is None:
        print(f"[VectorStore] Loading embedding model: {model_name}...")
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            trust_remote_code=True,
        )
        print(f"[VectorStore] ✓ Embedding model loaded")
    return _embedding_fn


def get_chroma_client(persist_dir: str = None) -> chromadb.PersistentClient:
    """Get or create the ChromaDB client (singleton)."""
    global _chroma_client
    if _chroma_client is None:
        if persist_dir is None:
            # Default: project_root/chroma_db
            persist_dir = str(Path(__file__).resolve().parent.parent.parent / "chroma_db")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
        print(f"[VectorStore] ✓ ChromaDB initialized at {persist_dir}")
    return _chroma_client


def get_collection(collection_name: str = "documents"):
    """Get or create the document collection."""
    client = get_chroma_client()
    ef = get_embedding_function()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_document(doc_structure: DocumentStructure, collection_name: str = "documents"):
    """
    Index all elements of a document into ChromaDB.
    Each element becomes a separate document in the collection.
    """
    collection = get_collection(collection_name)
    
    ids = []
    documents = []
    metadatas = []
    
    for element in doc_structure.elements:
        # ChromaDB ID must be unique globally
        chroma_id = f"{doc_structure.doc_id}__{element.id}"
        
        # Use contextual enrichment for embedding (better semantic search)
        # Falls back to raw content if context was not built
        if element.context:
            search_text = element.context
        elif element.type == "table_cell":
            search_text = f"[{element.table_id} Row {element.row} Col {element.col}] {element.content}"
        else:
            search_text = element.content
        
        if not element.content.strip():
            continue
        
        ids.append(chroma_id)
        documents.append(search_text)
        metadatas.append({
            "doc_id": doc_structure.doc_id,
            "file_name": doc_structure.file_name,
            "element_id": element.id,
            "element_type": element.type,
            "table_id": element.table_id or "",
            "row": element.row if element.row is not None else -1,
            "col": element.col if element.col is not None else -1,
            "original_content": element.content,  # keep original for exact retrieval
        })
    
    if ids:
        # Add in batches of 100 (ChromaDB recommendation)
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
            )
        print(f"[VectorStore] ✓ Indexed {len(ids)} elements for '{doc_structure.file_name}' (doc_id: {doc_structure.doc_id})")
    
    return len(ids)


def search(query: str, doc_ids: list[str] = None, top_k: int = 10, collection_name: str = "documents") -> list[dict]:
    """
    Semantic search across documents.
    
    Args:
        query: search text
        doc_ids: optional list of doc_ids to filter by
        top_k: number of results
    
    Returns:
        List of dicts with: id, content, metadata, distance
    """
    collection = get_collection(collection_name)
    
    where_filter = None
    if doc_ids:
        if len(doc_ids) == 1:
            where_filter = {"doc_id": doc_ids[0]}
        else:
            where_filter = {"doc_id": {"$in": doc_ids}}
    
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_filter,
    )
    
    output = []
    if results and results['ids'] and results['ids'][0]:
        for i, chroma_id in enumerate(results['ids'][0]):
            output.append({
                "id": chroma_id,
                "content": results['documents'][0][i] if results['documents'] else "",
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                "distance": results['distances'][0][i] if results['distances'] else 0,
            })
    
    return output


def get_full_document(doc_id: str, collection_name: str = "documents") -> list[dict]:
    """
    Retrieve ALL elements of a document, ordered by element ID.
    
    Returns list of dicts with: id, content, metadata
    """
    collection = get_collection(collection_name)
    
    results = collection.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas"],
    )
    
    output = []
    if results and results['ids']:
        for i, chroma_id in enumerate(results['ids']):
            output.append({
                "id": chroma_id,
                "content": results['documents'][i] if results['documents'] else "",
                "metadata": results['metadatas'][i] if results['metadatas'] else {},
            })
    
    # Sort by element_id to maintain reading order
    def sort_key(item):
        eid = item.get('metadata', {}).get('element_id', '')
        # Extract numeric parts for proper sorting: "Para_0" → (0, 0, -1, -1), "Table_0_Cell_1_2" → (1, 0, 1, 2)
        if eid.startswith('Para_'):
            try:
                return (0, int(eid.split('_')[1]), -1, -1)
            except (IndexError, ValueError):
                return (0, 999, -1, -1)
        elif eid.startswith('Table_'):
            parts = eid.split('_')
            try:
                table_num = int(parts[1])
                row = int(parts[3]) if len(parts) > 3 else -1
                col = int(parts[4]) if len(parts) > 4 else -1
                return (1, table_num, row, col)
            except (IndexError, ValueError):
                return (1, 999, -1, -1)
        return (2, 0, -1, -1)
    
    output.sort(key=sort_key)
    return output


def get_document_text(doc_id: str, collection_name: str = "documents") -> str:
    """Get full document content as formatted text for LLM context."""
    elements = get_full_document(doc_id, collection_name)
    lines = []
    for el in elements:
        eid = el.get("metadata", {}).get("element_id", "?")
        content = el.get("content", "")
        lines.append(f"[{eid}] {content}")
    return "\n".join(lines)


def list_documents(collection_name: str = "documents") -> list[dict]:
    """List all unique documents in the collection."""
    collection = get_collection(collection_name)
    
    # Get all metadatas
    results = collection.get(include=["metadatas"])
    
    docs = {}
    if results and results['metadatas']:
        for meta in results['metadatas']:
            doc_id = meta.get('doc_id', '')
            if doc_id and doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "file_name": meta.get('file_name', 'unknown'),
                }
    
    return list(docs.values())


def delete_document(doc_id: str, collection_name: str = "documents"):
    """Remove all elements of a document from ChromaDB."""
    collection = get_collection(collection_name)
    
    # Get all IDs for this document
    results = collection.get(
        where={"doc_id": doc_id},
    )
    
    if results and results['ids']:
        collection.delete(ids=results['ids'])
        print(f"[VectorStore] ✓ Deleted {len(results['ids'])} elements for doc_id: {doc_id}")
        return len(results['ids'])
    
    return 0
