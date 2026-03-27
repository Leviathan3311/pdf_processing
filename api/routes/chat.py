"""
Chat route - Single unified endpoint for all LLM operations.

The LangChain Agent automatically determines the intent (Q&A, compare, 
summarize, edit) and routes to the appropriate tool.
"""
import re
from pathlib import Path
from fastapi import APIRouter, HTTPException

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from api.schemas import ChatRequest, ChatResponse
from llm_pipeline import llm_engine, tools, vector_store


router = APIRouter(tags=["Chat"])

# No global agent needed anymore


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Unified chat endpoint. Send a message and optionally specify doc_ids.
    
    The LLM Agent automatically determines the intent:
    - Q&A → uses chat_tool (RAG search + answer)
    - Compare → uses compare_tool (load 2 docs, diff)
    - Summarize → uses chat_tool (summarize prompt)
    - Edit → uses edit_tool (JSON modifications → doc surgery → .docx file)
    
    Examples:
    - {"message": "Tóm tắt nội dung file này", "doc_ids": ["id1"]}
    - {"message": "So sánh 2 file này", "doc_ids": ["id1", "id2"]}
    - {"message": "Sửa giá trong bảng 1 thành 50 triệu"}
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    # If specific doc_ids provided, temporarily scope tools to those docs
    if request.doc_ids:
        # Verify doc_ids exist
        for doc_id in request.doc_ids:
            if not tools.get_doc_info(doc_id):
                raise HTTPException(
                    status_code=404,
                    detail=f"Document not found: {doc_id}"
                )
    
    try:
        # Build context info for the agent
        available_docs = []
        doc_ids_to_use = request.doc_ids if request.doc_ids else tools.get_all_doc_ids()
        
        for doc_id in doc_ids_to_use:
            info = tools.get_doc_info(doc_id)
            if info:
                available_docs.append(f"- {info['file_name']} (ID: {doc_id})")
        
        docs_context = "\n".join(available_docs) if available_docs else "Chưa có tài liệu nào."
        
        # Enhance the user message with context about available documents
        enhanced_input = f"""Tài liệu hiện có:
{docs_context}

Yêu cầu của người dùng: {request.message}"""

        # Run the robust custom agent loop
        all_tools = tools.get_all_tools()
        result = llm_engine.run_agent(enhanced_input, tools=all_tools, max_steps=4)
        
        response_text = result.get("output", "Không có kết quả.")
        generated_files = result.get("files", [])
        
        return ChatResponse(
            response=response_text,
            files=generated_files,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
