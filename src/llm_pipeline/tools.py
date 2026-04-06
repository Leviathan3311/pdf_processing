"""
LangChain Tools - 5 tools for the Agent.

1. chat_tool: Q&A + summarize (RAG-based) — auto-saves summary to session context
2. compare_tool: Compare two documents 
3. edit_tool: Modify document content → JSON output → doc surgery → .docx file
4. batch_rewrite_tool: Rewrite ENTIRE document with context from session
5. export_tool: Export uploaded document as Word or PDF for download
"""
import json
import re
from typing import Optional
import contextvars

from langchain_core.tools import tool

from llm_pipeline import vector_store
from llm_pipeline import llm_engine
from llm_pipeline import doc_surgery

# Context variable to hold doc_ids for the current API request
current_request_doc_ids = contextvars.ContextVar("current_request_doc_ids", default=None)

# Context variable to hold session_id for the current API request
current_session_id = contextvars.ContextVar("current_session_id", default="default")

# ─────────────────────────────────────────────────────────────
# Store for document metadata (doc_id → file_name, docx_path)
# Populated during upload. Shared across tools.
# ─────────────────────────────────────────────────────────────
_doc_registry: dict[str, dict] = {}

# ─────────────────────────────────────────────────────────────
# Session Context Store
# Stores important results (summaries, key info) per session.
# This allows batch_rewrite_tool to access accumulated context
# WITHOUT relying on the small LLM to pass it correctly.
# ─────────────────────────────────────────────────────────────
_session_contexts: dict[str, list[str]] = {}


def save_session_context(session_id: str, context: str):
    """Save important context (summary result, key info) for later use by rewrite tools.

    Only saves if the content looks like real document content (not a greeting,
    upload acknowledgement, or error message).  This prevents the session context
    from being polluted when the agent auto-responds to a file upload with a short
    acknowledgement phrase.
    """
    if not context or not context.strip():
        return

    # Don't save obviously non-content responses
    _NOISE_PHRASES = [
        "bạn muốn làm gì", "tôi đã nhận được", "tài liệu đã được tải",
        "bạn cần tôi làm gì", "tôi có thể giúp gì",
        "xin chào", "vui lòng upload", "chưa có tài liệu",
        "lỗi", "error",
    ]
    ctx_lower = context.strip().lower()
    if any(phrase in ctx_lower for phrase in _NOISE_PHRASES):
        print(f"[Session Context] ⏭️ Skipped saving noise/greeting message for session '{session_id}'")
        return
    # Also skip very short responses (< 80 chars) — likely not real document content
    if len(context.strip()) < 80:
        print(f"[Session Context] ⏭️ Skipped saving short response ({len(context.strip())} chars)")
        return

    if session_id not in _session_contexts:
        _session_contexts[session_id] = []
    # Keep only the last 5 contexts to avoid memory bloat
    _session_contexts[session_id].append(context.strip())
    if len(_session_contexts[session_id]) > 5:
        _session_contexts[session_id] = _session_contexts[session_id][-5:]
    print(f"[Session Context] ✓ Saved context for session '{session_id}' ({len(context)} chars)")


def get_session_context(session_id: str, most_recent_only: bool = True) -> str:
    """Get session context for the given session.

    Args:
        session_id: The session to fetch context for.
        most_recent_only: When True (default), returns only the MOST RECENT saved
            context.  This is the safe default for batch_rewrite_tool: if the user
            summarised File A then uploaded File B and asked to rewrite it, we want
            to fill File B with File A's summary — not a concatenation of both
            summaries which would confuse the LLM.
            Pass most_recent_only=False only when you explicitly want the full history
            (e.g. for a compare or audit task).
    """
    contexts = _session_contexts.get(session_id, [])
    if not contexts:
        return ""
    if most_recent_only:
        return contexts[-1]
    return "\n\n---\n\n".join(contexts)


def _get_context_from_history(session_id: str) -> str:
    """
    Fallback: extract useful context from session history.
    Searches for the most recent assistant message that looks like a real
    summary/extraction result — skips upload acknowledgements, greetings,
    error messages, and anything that is NOT document content.
    """
    from llm_pipeline.llm_engine import _session_histories
    history = _session_histories.get(session_id, [])

    # Phrases that indicate the message is NOT a useful document summary
    _SKIP_PHRASES = [
        "lỗi", "error",
        "vui lòng",
        "chưa có tài liệu",
        "tool_call", "<tool_call>",
        # Upload acknowledgements / greetings — these are NOT summaries
        "bạn muốn làm gì",
        "tôi đã nhận được",
        "tài liệu đã được tải",
        "tài liệu vừa tải lên",
        "bạn cần tôi làm gì",
        "tôi có thể giúp gì",
        "xin chào", "hello",
    ]

    # Keywords that strongly suggest the message IS a real summary/extraction
    _CONTENT_SIGNALS = [
        "tổ chức", "đối tượng", "sự kiện", "thời gian", "địa điểm",  # event summaries
        "mục đích", "yêu cầu", "nội dung", "thông tin",
        "tóm tắt", "tổng hợp", "kết quả", "số liệu",
        "điều khoản", "hợp đồng", "bên a", "bên b",
    ]

    for msg in reversed(history):
        if msg["role"] != "assistant":
            continue
        content = msg["content"]
        if len(content) < 100:
            continue
        content_lower = content.lower()

        # Hard skip — clearly not a summary
        if any(phrase in content_lower for phrase in _SKIP_PHRASES):
            continue

        # Prefer messages that have at least one content signal
        if any(signal in content_lower for signal in _CONTENT_SIGNALS):
            return content

    # Second pass: no content signals found — return any non-skipped substantial message
    for msg in reversed(history):
        if msg["role"] != "assistant":
            continue
        content = msg["content"]
        if len(content) < 100:
            continue
        content_lower = content.lower()
        if any(phrase in content_lower for phrase in _SKIP_PHRASES):
            continue
        return content

    return ""


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
# Tool 1: Chat (Q&A + Summarize) — always saves result to session context
# ─────────────────────────────────────────────────────────────


@tool
def chat_tool(query: str) -> str:
    """Trả lời câu hỏi hoặc tóm tắt nội dung tài liệu đã upload.
    Dùng tool này khi người dùng hỏi về nội dung, yêu cầu tóm tắt, 
    hoặc cần thông tin từ tài liệu. Input là câu hỏi của người dùng."""
    
    # Priority: doc_ids from current request > all registered doc_ids
    req_docs = current_request_doc_ids.get()
    doc_ids = req_docs if req_docs is not None else get_all_doc_ids()
    
    if not doc_ids:
        return "Chưa có tài liệu nào được chỉ định để trả lời câu hỏi. Vui lòng kiểm tra lại tải lên."
    


    # First, try to fetch the FULL text for all selected documents
    # to avoid context fragmentation (especially useful for tables/statistics across multiple files).
    full_text_mode = True
    all_texts = []
    total_length = 0
    
    for doc_id in doc_ids:
        doc_info = get_doc_info(doc_id)
        if not doc_info:
            continue
        
        file_name = doc_info.get("file_name", "unknown")
        doc_text = vector_store.get_document_text(doc_id)
        
        doc_block = f"--- TÀI LIỆU: {file_name} ---\n{doc_text}\n"
        all_texts.append(doc_block)
        total_length += len(doc_block)
    
    # Safe limit: 60,000 characters (approx 15-20k tokens for Qwen3-4B which supports 32k)
    SAFE_CHAR_LIMIT = 60000
    
    if total_length > 0 and total_length <= SAFE_CHAR_LIMIT:
        print(f"[chat_tool] Using FULL TEXT extraction. Total length: {total_length} chars.")
        context = "\n".join(all_texts)
    else:
        # Fallback to RAG if documents are too large
        full_text_mode = False
        print(f"[chat_tool] Documents too large ({total_length} chars). Falling back to RAG.")
        
        # Calculate dynamic top_k per document to ensure all docs are represented
        if len(doc_ids) == 1:
            top_k_per_doc = 15
        elif len(doc_ids) <= 3:
            top_k_per_doc = 7
        elif len(doc_ids) <= 6:
            top_k_per_doc = 5
        else:
            top_k_per_doc = 3
            
        all_results = []
        for doc_id in doc_ids:
            results = vector_store.search(query, doc_ids=[doc_id], top_k=top_k_per_doc)
            all_results.extend(results)
            
        if not all_results:
            return "Không tìm thấy thông tin liên quan trong tài liệu."
        
        # Separate paragraphs and table cells for proper reconstruction
        para_parts = []
        table_cells_by_table = {}  # (file_name, table_id) -> {(row, col): content}
        table_max_dims = {}  # (file_name, table_id) -> (max_row, max_col)
        
        for r in all_results:
            meta = r.get("metadata", {})
            file_name = meta.get("file_name", "unknown")
            element_id = meta.get("element_id", "?")
            element_type = meta.get("element_type", "")
            content = meta.get("original_content", "") or r.get("content", "")
            
            if element_type == "table_cell":
                table_id = meta.get("table_id", "")
                row = meta.get("row", 0)
                col = meta.get("col", 0)
                key = (file_name, table_id)
                
                if key not in table_cells_by_table:
                    table_cells_by_table[key] = {}
                    table_max_dims[key] = (-1, -1)
                
                table_cells_by_table[key][(row, col)] = content
                mr, mc = table_max_dims[key]
                table_max_dims[key] = (max(mr, row), max(mc, col))
            else:
                para_parts.append(f"[{file_name} - {element_id}] {content}")
        
        # Reconstruct table cells into Markdown tables
        for (file_name, table_id), cells in table_cells_by_table.items():
            max_row, max_col = table_max_dims[(file_name, table_id)]
            para_parts.append(f"\n[{file_name} - {table_id}] Bảng dữ liệu (trích xuất một phần):")
            for row in range(max_row + 1):
                row_parts = []
                for col in range(max_col + 1):
                    cell_text = cells.get((row, col), "...")
                    cell_text = cell_text.replace("\n", " ").strip()
                    row_parts.append(cell_text)
                para_parts.append("| " + " | ".join(row_parts) + " |")
                if row == 0:
                    para_parts.append("|" + "|".join(["---"] * (max_col + 1)) + "|")
            
        context = "\n".join(para_parts)
    
    prompt = f"""Dựa trên nội dung tài liệu dưới đây, hãy trả lời câu hỏi.

QUY TẮC BẮT BUỘC:
1. CHỈ sử dụng thông tin CÓ TRONG tài liệu bên dưới. TUYỆT ĐỐI KHÔNG bịa thêm hay tự sáng tạo dữ liệu.
2. Khi trích dẫn dữ liệu bảng, phải SAO CHÉP CHÍNH XÁC từng giá trị ô (cell) từ bảng Markdown, giữ nguyên tên người, địa chỉ, số liệu, không được thay đổi hay tóm tắt lại.
3. Nếu tài liệu chứa bảng dạng Markdown (dùng ký tự | và ---), hãy đọc theo hàng và cột, mỗi ô phân cách bởi dấu |.
4. Nếu thông tin không có trong tài liệu, trả lời: "Thông tin này không có trong tài liệu."

=== NỘI DUNG TÀI LIỆU ===
{context}

=== CÂU HỎI ===
{query}

=== TRẢ LỜI ==="""
    
    result = llm_engine.generate_raw(prompt)
    
    # ── Always save the result to session context so batch_rewrite_tool
    # can reliably retrieve it later, regardless of whether the user's query
    # contained explicit summary keywords.
    # Previously this only fired when _SUMMARY_KEYWORDS matched the query string
    # that the LLM passed when calling the tool — which was fragile because the
    # LLM might call chat_tool with a paraphrased instruction that doesn't contain
    # the exact keywords, silently skipping the save and causing batch_rewrite_tool
    # to fall back to the wrong history message.
    session_id = current_session_id.get()
    save_session_context(session_id, result)
    print(f"[chat_tool] 📝 Auto-saved result as session context (session: {session_id}, {len(result)} chars)")
    
    return result


# ─────────────────────────────────────────────────────────────
# Tool 2: Compare
# ─────────────────────────────────────────────────────────────

@tool
def compare_tool(input_text: str) -> str:
    """So sánh hai tài liệu đã upload và liệt kê các điểm khác biệt.
    Dùng tool này khi người dùng yêu cầu so sánh, đối chiếu hai file.
    Input là yêu cầu so sánh của người dùng (ví dụ: 'so sánh giá giữa 2 file')."""
    
    req_docs = current_request_doc_ids.get()
    doc_ids = req_docs if req_docs is not None else get_all_doc_ids()
    
    if len(doc_ids) < 2:
        return "Cần ít nhất 2 tài liệu để so sánh. Vui lòng kiểm tra lại tải lên."
    
    # Get full content of the two most recent documents in the scoped context
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
    
    req_docs = current_request_doc_ids.get()
    doc_ids = req_docs if req_docs is not None else get_all_doc_ids()
    
    if not doc_ids:
        return "Chưa có tài liệu nào được chỉ định để sửa đổi. Vui lòng kiểm tra lại tải lên."
    
    # Use the most recently uploaded document in the scoped context
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


# ─────────────────────────────────────────────────────────────
# Tool 4: Batch Rewrite — Two-phase approach
#
# Problem: Qwen3-4B is too small to understand element-to-context
# mapping in JSON format. It copies original text instead of replacing.
#
# Solution: Two-phase approach:
#   Phase 1: Programmatically classify elements as structural/content
#   Phase 2: Ask LLM to rewrite ONLY content elements (simpler task)
# ─────────────────────────────────────────────────────────────

# Patterns that identify structural/template elements (should NOT be rewritten)
_STRUCTURAL_PATTERNS = re.compile(
    r"(?i)("
    r"cộng\s*ho[àa]\s*x[ãa]\s*h[ộo]i|"  # CỘNG HOÀ XÃ HỘI
    r"độc\s*lập|tự\s*do|hạnh\s*phúc|"     # Độc lập - Tự do - Hạnh phúc
    r"kính\s*gửi|"                          # Kính gửi
    r"nơi\s*nhận|"                          # Nơi nhận
    r"ngày.*tháng.*năm|"                    # ngày...tháng...năm
    r"^(tl\.|kt\.|q\.|tm\.)|"              # TL. KT. Q. TM. (signatures)
    r"phó\s*(giám\s*đốc|chủ\s*tịch|trưởng)|" # Phó Giám đốc etc
    r"giám\s*đốc|chủ\s*tịch|trưởng\s*phòng|" # Giám đốc etc
    r"ký\s*tên|chữ\s*ký|"                  # Ký tên, chữ ký
    r"v/v:|số:|"                            # V/v:, Số:
    r"^\s*$"                                # Empty
    r")",
    re.UNICODE
)

def _word_overlap_ratio(text1: str, text2: str) -> float:
    """Calculate word-level Jaccard similarity between two texts.
    Used to detect when LLM partially copied original content instead of rewriting.
    Returns 0.0–1.0; values above ~0.35 indicate significant overlap with original.
    """
    if not text1 or not text2:
        return 0.0
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


def _classify_element(element: dict, index: int, total_elements: int, noi_nhan_idx: int) -> str:
    """
    Classify an element as 'structural' (keep) or 'content' (rewrite).
    
    Structural: headers, company names, signatures, dates, section numbers
    Content: body paragraphs with actual report/document content
    """
    meta = element.get("metadata", {})
    eid = meta.get("element_id", "")
    etype = meta.get("element_type", "")
    content = (meta.get("original_content", "") or element.get("content", "")).strip()
    
    # Explicit structural patterns
    if _STRUCTURAL_PATTERNS.search(content):
        return "structural"
    
    # Very short noise content
    if len(content) < 10:
        return "structural"
        
    is_header_region = index < 15
    is_footer_region = (noi_nhan_idx != -1 and index >= noi_nhan_idx) or (index > total_elements - 5)
    
    if is_header_region or is_footer_region:
        if etype == "table_cell" and (eid.startswith("Table_0_Cell_") or eid.startswith("Table_1_Cell_")):
            return "structural"
        if content == content.upper() and len(content) < 100:
            return "structural"
            
    if is_footer_region:
        if len(content) < 40:
            return "structural"
            
    # Everything else in the middle of the document is content that should be rewritten
    return "content"


@tool
def batch_rewrite_tool(instruction: str, context: str = "") -> str:
    """Viết lại (rewrite) TOÀN BỘ nội dung file tự động theo lô (batch).
    Dùng công cụ này KHI người dùng yêu cầu viết lại toàn bộ file, sửa nội dung theo tóm tắt, hoặc xuất nội dung theo mẫu file.
    Tham số `context` CẦN chứa toàn bộ thông tin/tổng hợp mà bạn muốn dùng để viết lại. Nếu để trống, hệ thống sẽ tự động lấy từ lịch sử hội thoại.
    Tham số `instruction` là yêu cầu cụ thể (vd: 'Viết lại hợp đồng theo nội dung tóm tắt')."""
    
    req_docs = current_request_doc_ids.get()
    doc_ids = req_docs if req_docs is not None else get_all_doc_ids()
    
    if not doc_ids:
        return "Chưa có tài liệu nào được chỉ định để sửa đổi toàn bộ. Vui lòng kiểm tra lại tải lên."
    
    # ── Auto-fetch context if the LLM passed an empty/inadequate context ──
    session_id = current_session_id.get()
    
    context_from_session = False  # Track whether context was auto-fetched (→ always template mode)

    if not context or not context.strip() or len(context.strip()) < 50:
        print(f"[batch_rewrite_tool] ⚠️ Context parameter is empty/short. Auto-fetching from session '{session_id}'...")
        
        # Priority 1: Saved session context (summaries from chat_tool)
        saved_context = get_session_context(session_id)
        if saved_context and len(saved_context) > 50:
            context = saved_context
            context_from_session = True
            print(f"[batch_rewrite_tool] ✓ Using saved session context ({len(context)} chars)")
        else:
            # Priority 2: Extract from session history
            history_context = _get_context_from_history(session_id)
            if history_context:
                context = history_context
                context_from_session = True
                print(f"[batch_rewrite_tool] ✓ Using history context ({len(context)} chars)")
            else:
                return ("Lỗi: Không tìm thấy thông tin tổng hợp/tóm tắt nào trong phiên hội thoại. "
                        "Vui lòng tóm tắt hoặc cung cấp nội dung trước, sau đó yêu cầu viết lại.")
    
    print(f"[batch_rewrite_tool] Context preview (first 300 chars): {context[:300]}...")
    
    # Lấy tài liệu mới nhất (chính là template format) trong scoped context
    doc_id = doc_ids[-1]
    doc_info = get_doc_info(doc_id)
    if not doc_info:
        return "Không tìm thấy thông tin tài liệu."
    
    # Lấy toàn bộ elements nối tiếp nhau
    elements = vector_store.get_full_document(doc_id)
    if not elements:
        return "Tài liệu trống."
        
    docx_path = doc_info.get("docx_path", "")
    
    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Classify elements as structural vs content
    # ══════════════════════════════════════════════════════════════
    valid_elements = [el for el in elements if el.get("content", "").strip()]
    
    noi_nhan_idx = -1
    for i, el in enumerate(valid_elements):
        content = (el.get("metadata", {}).get("original_content", "") or el.get("content", "")).strip().lower()
        if "nơi nhận" in content:
            noi_nhan_idx = i
            break
            
    structural_elements = []  # Keep as-is (headers, signatures, dates...)
    content_elements = []     # Rewrite with context
    
    total_elements = len(valid_elements)
    
    # ── MAPPING BODY REGION ──
    # We find the FIRST and LAST element classified as "content".
    # EVERYTHING in between must be considered "content" (to be wiped),
    # so we don't accidentally leave old section headers (like "I. ĐẶC ĐIỂM")
    # mixed in our newly inserted summary.
    classifications = []
    first_content_idx = -1
    last_content_idx = -1
    
    for i, el in enumerate(valid_elements):
        cls = _classify_element(el, i, total_elements, noi_nhan_idx)
        classifications.append(cls)
        if cls == "content":
            if first_content_idx == -1:
                first_content_idx = i
            last_content_idx = i
            
    if first_content_idx == -1:
        return "Không tìm thấy đoạn nội dung nào cần viết lại trong tài liệu mẫu."
        
    for i, el in enumerate(valid_elements):
        meta = el.get("metadata", {})
        eid = meta.get("element_id", "?")
        content = (meta.get("original_content", "") or el.get("content", "")).strip()
        
        # Override classification: if it's within the body boundaries, it's body!
        if first_content_idx <= i <= last_content_idx:
            content_elements.append(el)
            print(f"  [REWRITE/BODY] {eid}: {content[:60]}...")
        else:
            structural_elements.append(el)
            print(f"  [KEEP/STRUCT]  {eid}: {content[:60]}...")
            
    print(f"\n[Batch Rewrite] Structural (keep): {len(structural_elements)}, Content (rewrite & wipe): {len(content_elements)}")
    
    # ── Replace Body Mode ──
    # Always use replace_body mode since this tool is specifically for
    # completely rewriting the document, and the user explicitly requested
    # creating a new document with same structural format but new content.
    print(f"[Batch Rewrite] ⚡ Chế độ Replace Body: Yêu cầu LLM viết lại nội dung tự nhiên, sau đó map vào file mẫu.")
    
    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Rewrite Strategy
    # ══════════════════════════════════════════════════════════════
    # Gather original text to pass to LLM as style reference
    original_text_parts = []
    for el in content_elements:
        content = (el.get("metadata", {}).get("original_content", "") or el.get("content", "")).strip()
        if content:
            original_text_parts.append(content)
            
    original_text = "\n".join(original_text_parts)
    # Target length ~8000 chars to save tokens while giving enough style context
    if len(original_text) > 8000:
        original_text = original_text[:8000] + "\n...[lược bớt]..."

    prompt = f"""Bạn là một chuyên gia soạn thảo văn bản hành chính.
Nhiệm vụ của bạn là viết lại phần thân của một văn bản hoàn chỉnh, tự nhiên và chuyên nghiệp dựa trên BẢN TÓM TẮT.
Tuy nhiên, bạn PHẢI BẮT CHƯỚC VĂN PHONG, cách trình bày (có chia mục, gạch đầu dòng, giọng điệu...) của VĂN BẢN MẪU.

=== VĂN BẢN MẪU (THAM KHẢO VĂN PHONG VÀ CẤU TRÚC) ===
{original_text}

=== BẢN TÓM TẮT MỚI (Dữ liệu nội dung cần được đưa vào) ===
{context}

=== YÊU CẦU CỦA NGƯỜI DÙNG ===
{instruction}

=== QUY TẮC BẮT BUỘC ===
1. CHỈ viết nội dung phần thân của văn bản (KHÔNG viết tiêu đề, KHÔNG viết quốc hiệu, KHÔNG viết chữ ký hay nơi nhận).
2. Nội dung chính PHẢI bám sát BẢN TÓM TẮT MỚI. Tuyệt đối KHÔNG giữ lại các sự kiện/số liệu của VĂN BẢN MẪU nếu Bản Tóm Tắt Mới không nhắc đến.
3. CÁCH HÀNH VĂN, cách chia mục (I, II, 1, 2...), cách đặt câu chữ chuyên ngành phải GIỐNG VỚI VĂN BẢN MẪU.
4. KHÔNG sử dụng các thẻ [ID] hay vẽ lại bảng. Chỉ trình bày nội dung bằng văn bản thuần túy (các đoạn cách nhau khoảng trắng).

BẮT ĐẦU VIẾT VĂN BẢN:
"""
    response = llm_engine.generate_raw(prompt, max_new_tokens=4096)
    
    import re
    # Loại bỏ các block code markdown (nếu LLM có in thừa)
    response = re.sub(r"```[^\n]*\n(.*?)```", r"\1", response, flags=re.DOTALL)
    
    # Chia nhỏ đoạn văn bản LLM sinh ra thành các đoạn văn riêng biệt
    summary_paras = [p.strip() for p in response.split('\n') if p.strip()]
    
    if not summary_paras:
        print("[Batch Rewrite] ⚠️ LLM trả về rỗng, dùng fallback context...")
        summary_paras = [p.strip() for p in context.split('\n') if p.strip()]

    # Collect all content element IDs
    content_eids = [el.get("metadata", {}).get("element_id", "?") for el in content_elements if "element_id" in el.get("metadata", {})]
    
    modifications = {
        "action": "replace_body",
        "content_eids": content_eids,
        "new_paragraphs": summary_paras
    }
        
    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Apply modifications via doc surgery
    # ══════════════════════════════════════════════════════════════
    try:
        from pathlib import Path
        output_dir = Path(docx_path).parent.parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        revised_path = doc_surgery.apply_modifications(
            docx_path=docx_path,
            modifications=modifications,
            output_dir=str(output_dir),
        )
        
        return f"✓ Đã viết lại toàn bộ tài liệu thành công. File mới: {revised_path}"
    except Exception as e:
        return f"Lỗi khi áp dụng sửa đổi: {str(e)}"


# ─────────────────────────────────────────────────────────────
# Tool 5: Export (download Word/PDF file)
# ─────────────────────────────────────────────────────────────

@tool
def export_tool(format: str = "docx") -> str:
    """Xuất file tài liệu đã upload hoặc đã chỉnh sửa dưới dạng Word hoặc PDF để tải về.
    Dùng tool này khi người dùng yêu cầu xuất file, tải file, download, convert sang word/pdf.
    Tham số format: 'docx' cho file Word, 'pdf' cho file PDF. Mặc định là 'docx'."""
    
    req_docs = current_request_doc_ids.get()
    doc_ids = req_docs if req_docs is not None else get_all_doc_ids()
    
    if not doc_ids:
        return "Chưa có tài liệu nào được chỉ định để xuất. Vui lòng tải lên tài liệu trước."
    
    # Use the most recently uploaded document
    doc_id = doc_ids[-1]
    doc_info = get_doc_info(doc_id)
    
    if not doc_info:
        return "Không tìm thấy thông tin tài liệu."
    
    docx_path = doc_info.get("docx_path", "")
    file_name = doc_info.get("file_name", "document")
    
    if not docx_path:
        return "Không tìm thấy file DOCX của tài liệu này."
    
    from pathlib import Path
    from llm_pipeline import exporter
    
    output_dir = Path(docx_path).parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if there's a revised version available
    revised_path = output_dir / f"{Path(docx_path).stem}_Revised.docx"
    source_path = str(revised_path) if revised_path.exists() else docx_path
    source_name = Path(source_path).stem
    
    format_lower = format.strip().lower()
    
    if format_lower == "pdf":
        # Export as PDF
        pdf_filename = f"{source_name}.pdf"
        result = exporter.export_pdf(source_path, str(output_dir), pdf_filename)
        if result:
            return f"✓ Đã xuất file PDF thành công. File: {result}"
        else:
            return "Lỗi: Không thể xuất file PDF. Cần cài đặt MS Word hoặc LibreOffice."
    else:
        # Export as DOCX (copy to outputs)
        docx_filename = f"{source_name}_Export.docx"
        result = exporter.export_docx(source_path, str(output_dir), docx_filename)
        return f"✓ Đã xuất file Word thành công. File: {result}"


def get_all_tools() -> list:
    """Return all tools for agent registration."""
    return [chat_tool, compare_tool, edit_tool, batch_rewrite_tool, export_tool]