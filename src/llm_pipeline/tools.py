"""
LangChain Tools - 3 tools for the Agent.

1. chat_tool: Q&A + summarize (RAG-based)
2. compare_tool: Compare two documents 
3. edit_tool: Modify document content → JSON output → doc surgery → .docx file
"""
import json
from typing import Optional

from langchain_core.tools import tool

from llm_pipeline import vector_store
from llm_pipeline import llm_engine
from llm_pipeline import doc_surgery


# ─────────────────────────────────────────────────────────────
# Store for document metadata (doc_id → file_name, docx_path)
# Populated during upload. Shared across tools.
# ─────────────────────────────────────────────────────────────
_doc_registry: dict[str, dict] = {}


def register_document(doc_id: str, file_name: str, docx_path: str):
    """Register a document for tool access."""
    _doc_registry[doc_id] = {
        "file_name": file_name,
        "docx_path": docx_path,
    }


def get_doc_info(doc_id: str) -> Optional[dict]:
    """Get document info from registry."""
    return _doc_registry.get(doc_id)


def get_all_doc_ids() -> list[str]:
    """Get all registered document IDs."""
    return list(_doc_registry.keys())


# ─────────────────────────────────────────────────────────────
# Tool 1: Chat (Q&A + Summarize)
# ─────────────────────────────────────────────────────────────

@tool
def chat_tool(query: str) -> str:
    """Trả lời câu hỏi hoặc tóm tắt nội dung tài liệu đã upload.
    Dùng tool này khi người dùng hỏi về nội dung, yêu cầu tóm tắt, 
    hoặc cần thông tin từ tài liệu. Input là câu hỏi của người dùng."""
    
    doc_ids = get_all_doc_ids()
    if not doc_ids:
        return "Chưa có tài liệu nào được upload. Vui lòng upload tài liệu trước."
    
    # RAG: search relevant chunks
    # Giảm top_k xuống 7 (thay vì 15) để giảm lượng context đưa vào LLM, giúp mô hình xử lý nhanh hơn nhiều
    results = vector_store.search(query, doc_ids=doc_ids, top_k=7)
    
    if not results:
        return "Không tìm thấy thông tin liên quan trong tài liệu."
    
    # Build context
    context_parts = []
    for r in results:
        meta = r.get("metadata", {})
        file_name = meta.get("file_name", "unknown")
        element_id = meta.get("element_id", "?")
        content = r.get("content", "")
        context_parts.append(f"[{file_name} - {element_id}] {content}")
    
    context = "\n".join(context_parts)
    
    prompt = f"""Dựa trên nội dung tài liệu dưới đây, hãy trả lời câu hỏi.
Chỉ trả lời dựa trên thông tin có trong tài liệu, không bịa thêm.

=== NỘI DUNG TÀI LIỆU ===
{context}

=== CÂU HỎI ===
{query}

=== TRẢ LỜI ==="""
    
    return llm_engine.generate_raw(prompt)


# ─────────────────────────────────────────────────────────────
# Tool 2: Compare
# ─────────────────────────────────────────────────────────────

@tool
def compare_tool(input_text: str) -> str:
    """So sánh hai tài liệu đã upload và liệt kê các điểm khác biệt.
    Dùng tool này khi người dùng yêu cầu so sánh, đối chiếu hai file.
    Input là yêu cầu so sánh của người dùng (ví dụ: 'so sánh giá giữa 2 file')."""
    
    doc_ids = get_all_doc_ids()
    if len(doc_ids) < 2:
        return "Cần ít nhất 2 tài liệu để so sánh. Vui lòng upload thêm tài liệu."
    
    # Get full content of the two most recent documents
    doc1_id = doc_ids[-2]
    doc2_id = doc_ids[-1]
    
    doc1_info = get_doc_info(doc1_id)
    doc2_info = get_doc_info(doc2_id)
    
    doc1_text = vector_store.get_document_text(doc1_id)
    doc2_text = vector_store.get_document_text(doc2_id)
    
    doc1_name = doc1_info.get("file_name", "File 1") if doc1_info else "File 1"
    doc2_name = doc2_info.get("file_name", "File 2") if doc2_info else "File 2"
    
    prompt = f"""So sánh hai tài liệu sau và liệt kê các điểm khác biệt chi tiết.

=== TÀI LIỆU 1: {doc1_name} ===
{doc1_text}

=== TÀI LIỆU 2: {doc2_name} ===
{doc2_text}

=== YÊU CẦU ===
{input_text}

Hãy liệt kê các điểm khác biệt dưới dạng danh sách rõ ràng, bao gồm:
- Khác biệt về nội dung
- Khác biệt về số liệu/giá cả (nếu có)
- Khác biệt về cấu trúc
- Các điều khoản thêm/bớt (nếu có)

=== KẾT QUẢ SO SÁNH ==="""
    
    return llm_engine.generate_raw(prompt)


# ─────────────────────────────────────────────────────────────
# Tool 3: Edit (Modify document → JSON → doc surgery → .docx)
# ─────────────────────────────────────────────────────────────

@tool
def edit_tool(instruction: str) -> str:
    """Sửa đổi nội dung tài liệu theo yêu cầu, giữ nguyên format gốc.
    Dùng tool này khi người dùng yêu cầu sửa, thay đổi, cập nhật nội dung 
    (ví dụ: 'sửa giá thành 50 triệu', 'đổi tên công ty thành ABC').
    Input là yêu cầu sửa đổi cụ thể."""
    
    doc_ids = get_all_doc_ids()
    if not doc_ids:
        return "Chưa có tài liệu nào được upload. Vui lòng upload tài liệu trước."
    
    # Use the most recently uploaded document
    doc_id = doc_ids[-1]
    doc_info = get_doc_info(doc_id)
    
    if not doc_info:
        return "Không tìm thấy thông tin tài liệu."
    
    # Dùng vector_store để tìm các đoạn văn bản liên quan đến yêu cầu sửa đổi
    # Việc này giúp tránh load toàn bộ doc_text gây tràn RAM (OOM) cho GPU
    results = vector_store.search(instruction, doc_ids=[doc_id], top_k=20)
    
    if not results:
        return "Không tìm thấy phần nội dung nào trong tài liệu khớp với yêu cầu sửa đổi."
        
    doc_text_parts = []
    for r in results:
        meta = r.get("metadata", {})
        element_id = meta.get("element_id", "?")
        content = r.get("content", "")
        doc_text_parts.append(f"[{element_id}] {content}")
        
    doc_text = "\n".join(doc_text_parts)
    
    docx_path = doc_info.get("docx_path", "")
    file_name = doc_info.get("file_name", "document")
    
    # Generate JSON modifications using LLM
    prompt = f"""Bạn là hệ thống sửa đổi tài liệu. Dựa trên nội dung tài liệu và yêu cầu sửa đổi,
hãy trả về một JSON object chứa danh sách các thay đổi cần thực hiện.

=== NỘI DUNG TÀI LIỆU (mỗi dòng có format [ID] nội dung) ===
{doc_text}

=== YÊU CẦU SỬA ĐỔI ===
{instruction}

=== QUY TẮC ===
1. Chỉ sửa những phần được yêu cầu, giữ nguyên phần còn lại.
2. Trả về ĐÚNG format JSON sau, KHÔNG giải thích thêm:

```json
{{
  "modifications": [
    {{"id": "element_id ở đây", "new_text": "nội dung mới ở đây"}},
    {{"id": "element_id khác", "new_text": "nội dung mới khác"}}
  ]
}}
```

3. "id" phải là ID chính xác từ tài liệu (ví dụ: "Para_0", "Table_0_Cell_1_2").
4. "new_text" là nội dung mới thay thế cho element đó.

=== JSON OUTPUT ==="""
    
    response = llm_engine.generate_raw(prompt, max_new_tokens=4096)
    
    # Parse JSON from response
    try:
        # Try to extract JSON from response (might be wrapped in code block)
        json_str = response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        
        modifications = json.loads(json_str)
    except (json.JSONDecodeError, IndexError) as e:
        return f"Lỗi: LLM trả về JSON không hợp lệ. Vui lòng thử lại.\nResponse: {response[:500]}"
    
    # Apply modifications via doc surgery
    try:
        from pathlib import Path
        output_dir = Path(docx_path).parent.parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        revised_path = doc_surgery.apply_modifications(
            docx_path=docx_path,
            modifications=modifications,
            output_dir=str(output_dir),
        )
        
        return f"✓ Đã sửa đổi tài liệu thành công. File mới: {revised_path}"
    except Exception as e:
        return f"Lỗi khi áp dụng sửa đổi: {str(e)}"


def get_all_tools() -> list:
    """Return all tools for agent registration."""
    return [chat_tool, compare_tool, edit_tool]
