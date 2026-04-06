"""
LLM Engine - Load Qwen3-4B and create LangChain Agent.

Handles model loading (RTX 5070 Ti 16GB VRAM),
and creates a ReAct Agent with tools for chat, compare, and edit.
"""
import torch
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import re
from transformers import pipeline as hf_pipeline


# Singleton instances
_model = None
_tokenizer = None
_langchain_llm = None
_agent_executor = None
_session_histories = {}


def _safe_cuda_cleanup():
    """Safely clean CUDA cache, ignoring errors from corrupted CUDA state."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[LLM Engine] ⚠️ CUDA cleanup warning (non-fatal): {e}")


SYSTEM_PROMPT = """Bạn là trợ lý thông minh chuyên xử lý tài liệu. Bạn có thể:
1. Trả lời câu hỏi về nội dung tài liệu đã upload
2. So sánh hai tài liệu và liệt kê điểm khác biệt
3. Sửa đổi nội dung tài liệu theo yêu cầu (giữ nguyên format gốc)

Luôn trả lời bằng tiếng Việt trừ khi được yêu cầu khác.
Khi trả lời, hãy dựa trên nội dung thực tế từ tài liệu, không bịa thêm.

Bạn có các công cụ sau:
{tools}

Sử dụng format sau:

Question: câu hỏi đầu vào
Thought: suy nghĩ về việc cần làm
Action: tên tool cần dùng
Action Input: input cho tool
Observation: kết quả từ tool
... (lặp lại Thought/Action/Action Input/Observation nếu cần)
Thought: tôi đã có câu trả lời cuối cùng
Final Answer: câu trả lời cuối cùng cho người dùng

Danh sách tool: {tool_names}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


# ─────────────────────────────────────────────────────────────
# Intent Classification
# ─────────────────────────────────────────────────────────────

# Keyword fast-path: các pattern rõ ràng là casual chat
_CASUAL_PATTERNS = [
    # Chào hỏi
    r"\b(hello|hi|hey|xin chào|chào|alo)\b",
    # Hỏi thăm
    r"\b(bạn khỏe|bạn có khỏe|khỏe không|sao rồi|thế nào)\b",
    # Cảm ơn / tạm biệt
    r"\b(cảm ơn|camon|thanks|thank you|tạm biệt|bye|goodbye)\b",
    # Khen ngợi
    r"\b(tốt lắm|giỏi lắm|hay lắm|tuyệt|awesome|great|good job|well done)\b",
    # Câu phiếm
    r"\b(bạn là ai|bạn tên gì|bạn làm được gì|bạn có thể làm gì|giới thiệu bản thân)\b",
    # Biểu cảm đơn giản
    r"^(ok|okay|oke|được|rồi|vâng|dạ|ừ|uhh?|hmm+|ah+|oh+|wow)[!?.]*$",
]

_CASUAL_REGEX = re.compile(
    "|".join(_CASUAL_PATTERNS),
    re.IGNORECASE | re.UNICODE,
)

# Keyword fast-path: các pattern rõ ràng là document task
_DOCUMENT_PATTERNS = [
    r"\b(tài liệu|file|hợp đồng|văn bản|tóm tắt|so sánh|sửa|chỉnh|edit|upload|tải lên)\b",
    r"\b(docx|pdf|xlsx|doc)\b",
    r"\b(bảng|điều khoản|điều kiện|giá|số liệu|thông tin trong|nội dung)\b",
    r"\b(summarize|summary|compare|extract|modify|rewrite)\b",
    r"\b(xuất|export|download|tải về|tải xuống|convert|chuyển đổi|theo mẫu|viết lại)\b",
]

_DOCUMENT_REGEX = re.compile(
    "|".join(_DOCUMENT_PATTERNS),
    re.IGNORECASE | re.UNICODE,
)


def classify_intent(query: str) -> str:
    """
    Classify user query intent into one of:
      - "casual"   : general conversation, greetings, small talk
      - "document" : document Q&A, editing, comparison tasks
      - "ambiguous": unclear, let the agent loop decide

    Uses keyword fast-path first to avoid extra GPU calls.
    Falls back to "ambiguous" for the agent loop to handle gracefully.
    """
    query_stripped = query.strip()

    # Fast path 1: obvious casual (short + matches casual patterns)
    if _CASUAL_REGEX.search(query_stripped):
        # Extra guard: if it also mentions document keywords, treat as ambiguous
        if not _DOCUMENT_REGEX.search(query_stripped):
            return "casual"

    # Fast path 2: obvious document task
    if _DOCUMENT_REGEX.search(query_stripped):
        return "document"

    # Fast path 3: very short queries with no document keywords → likely casual
    word_count = len(query_stripped.split())
    if word_count <= 4 and not _DOCUMENT_REGEX.search(query_stripped):
        return "casual"

    # Everything else → let agent loop handle it (model decides whether to call tools)
    return "ambiguous"


def _generate_casual_response(query: str, session_id: str) -> str:
    """
    Generate a natural casual response using generate_raw().
    Bypasses the tool-calling agent loop entirely.
    """
    global _session_histories

    history = _session_histories.get(session_id, [])

    # Build a compact history string (last 6 turns max)
    history_text = ""
    for msg in history[-12:]:
        role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""Bạn là trợ lý AI thân thiện, chuyên hỗ trợ xử lý tài liệu nhưng cũng có thể trò chuyện bình thường.
Trả lời tự nhiên, ngắn gọn, thân thiện. Không đề cập đến tool hay tài liệu trừ khi người dùng hỏi.
Luôn trả lời bằng tiếng Việt.

{f"Lịch sử hội thoại gần đây:{chr(10)}{history_text}" if history_text else ""}
Người dùng: {query}
Trợ lý:"""

    return generate_raw(prompt, max_new_tokens=512)


def get_model_path() -> str:
    """Get the default model path."""
    root = Path(__file__).resolve().parent.parent.parent
    local_path = root / "Qwen3-4B"
    if local_path.exists():
        return str(local_path)
    return "Qwen/Qwen3-4B"


def load_model(
    model_path: str = None,
    load_4bit: bool = True,
    load_8bit: bool = False,
) -> tuple:
    """
    Load Qwen3-4B model and tokenizer.
    
    Uses same GPU loading pattern as existing Qwen2.5-VL-3B in processs_pdf_to_docs.py.
    """
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    if model_path is None:
        model_path = get_model_path()
    
    print(f"[LLM Engine] Loading Qwen3-4B from '{model_path}'...")
    
    model_kwargs = {
        "device_map": "auto",
    }
    
    if load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        print("[LLM Engine] Using 4-bit quantization (NF4)")
    elif load_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        print("[LLM Engine] Using 8-bit quantization")
    else:
        model_kwargs["torch_dtype"] = torch.float16
        print("[LLM Engine] Using float16 (no quantization)")
    
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _tokenizer.padding_side = "left"  # Required for batched generation
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
        
    _model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    print("[LLM Engine] ✓ Model loaded successfully")
    return _model, _tokenizer


def get_langchain_llm(model_path: str = None, load_4bit: bool = True, load_8bit: bool = False):
    """Get or create a LangChain-compatible LLM wrapper."""
    global _langchain_llm
    
    if _langchain_llm is not None:
        return _langchain_llm
    
    model, tokenizer = load_model(model_path, load_4bit, load_8bit)
    
    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=4096,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
    )
    
    _langchain_llm = HuggingFacePipeline(pipeline=pipe)
    
    print("[LLM Engine] ✓ LangChain LLM wrapper created")
    return _langchain_llm


def run_agent(query: str, tools: list, max_steps: int = 3, session_id: str = "default", raw_user_message: str = None) -> dict:
    """
    Native Qwen3 tool-calling loop with intent-aware routing.

    Flow:
      1. classify_intent() → "casual" | "document" | "ambiguous"
      2. "casual"   → _generate_casual_response() via generate_raw(), skip agent loop
      3. "document" / "ambiguous" → full agent tool-calling loop

    The agent system prompt no longer forces tool calls, so for "ambiguous"
    queries the model can freely choose to respond directly or call a tool.
    """
    global _session_histories

    # ── Step 1: Intent classification (keyword fast-path, no GPU cost) ──
    # IMPORTANT: classify on raw_user_message (not augmented query which contains
    # document context like "Tài liệu hiện có:..." that would skew the classifier)
    text_to_classify = raw_user_message if raw_user_message else query
    intent = classify_intent(text_to_classify)
    print(f"[Agent] 🧭 Intent: '{intent}' for message: '{text_to_classify[:60]}...'" if len(text_to_classify) > 60 else f"[Agent] 🧭 Intent: '{intent}' for message: '{text_to_classify}'")

    # ── Step 2: Casual fast-path ──
    if intent == "casual":
        print("[Agent] 💬 Routing to casual chat (bypassing tool loop)")
        actual_query = raw_user_message if raw_user_message else query
        response = _generate_casual_response(actual_query, session_id)

        # Save to history
        user_msg_to_save = raw_user_message if raw_user_message else query
        if session_id not in _session_histories:
            _session_histories[session_id] = []
        _session_histories[session_id].append({"role": "user", "content": user_msg_to_save})
        _session_histories[session_id].append({"role": "assistant", "content": response})

        _safe_cuda_cleanup()
        return {"output": response, "files": []}

    # ── Step 3: Document / ambiguous → full agent loop ──
    model, tokenizer = load_model()
    
    # 1. Prepare native HuggingFace tool schemas
    hf_tools = []
    tool_map = {}
    for t in tools:
        tool_map[t.name] = t
        param_schema = {"type": "object", "properties": {"input_text": {"type": "string", "description": "Lệnh cụ thể"}}, "required": ["input_text"]}
        if hasattr(t, "args_schema") and t.args_schema:
            if hasattr(t.args_schema, "model_json_schema"):
                param_schema = t.args_schema.model_json_schema()
            elif hasattr(t.args_schema, "schema"):
                param_schema = t.args_schema.schema()
                
        hf_tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": param_schema
            }
        })

    # ── Updated system prompt: model decides whether to call tools ──
    system_prompt = """Bạn là trợ lý xử lý tài liệu thông minh, thân thiện.

KHI NÀO GỌI TOOL:
- Dùng `chat_tool` khi người dùng hỏi nội dung, yêu cầu tóm tắt, lập bảng, trích xuất, thống kê thông tin từ tài liệu đã upload.
- Dùng `compare_tool` khi người dùng yêu cầu so sánh 2 tài liệu.
- Dùng `edit_tool` khi người dùng yêu cầu sửa MỘT ĐIỂM CỤ THỂ trong tài liệu (ví dụ: đổi giá, sửa tên).
- Dùng `batch_rewrite_tool` khi:
  * Người dùng yêu cầu viết lại TOÀN BỘ tài liệu.
  * Người dùng yêu cầu "xuất nội dung theo mẫu" hoặc "sửa document theo tóm tắt".
  * Người dùng đã tóm tắt trước đó và bây giờ muốn áp dụng vào file mẫu.
  * Lưu ý: context sẽ được hệ thống tự động lấy từ lịch sử, bạn có thể để trống tham số `context`.
- Dùng `export_tool` khi người dùng yêu cầu xuất file, tải file, download file, hoặc convert sang Word/PDF.

KHI NÀO KHÔNG CẦN GỌI TOOL:
- Câu hỏi về khả năng của bạn, hướng dẫn sử dụng → trả lời thẳng.
- Người dùng chưa upload tài liệu mà hỏi thông tin chung → hướng dẫn họ upload trước.

QUY TẮC BẮT BUỘC:
1. Khi có tin nhắn thông báo "[Tài liệu đã tải lên:...]":
   - NẾU trước đó đã có yêu cầu xử lý (VD: "tóm tắt", "dịch") → tự động gọi tool thực hiện ngay.
   - NẾU chưa có yêu cầu nào → hỏi người dùng muốn làm gì với tài liệu vừa tải lên.
2. NẾU người dùng yêu cầu thao tác trên tài liệu nhưng "Tài liệu hiện có" đang là "Chưa có tài liệu nào." → Báo lỗi thân thiện và hướng dẫn upload.
3. TUYỆT ĐỐI KHÔNG được chỉ mô tả "tôi sẽ gọi tool" hay "tôi cần gọi tool" — bạn PHẢI thực sự gọi tool ngay lập tức. Không hỏi lại, không giải thích trước, HÃY GỌI TOOL NGAY.
4. Khi người dùng nói "xuất theo mẫu", "sửa theo tóm tắt", "viết lại theo nội dung đã tóm tắt" → LUÔN gọi `batch_rewrite_tool`.
5. Khi người dùng nói "xuất word", "xuất pdf", "tải file", "download" → LUÔN gọi `export_tool`.

Khi trả lời dạng bảng thống kê, dùng format bảng Markdown (| và ---).
Luôn trả lời bằng tiếng Việt, rõ ràng, ngắn gọn."""

    if session_id not in _session_histories:
        _session_histories[session_id] = []

    messages = [{"role": "system", "content": system_prompt}]
    
    # Append the last N history messages to prevent context overflow (e.g. max 10 turns)
    messages.extend(_session_histories[session_id][-20:])

    # ── Context injection: if the user is asking to rewrite/export based on a
    # previous summary, embed that summary directly into the query so the agent
    # (and ultimately batch_rewrite_tool) receives it even if it forgets to pass
    # context= correctly.  This is the most robust fix for the "viết sai" bug.
    _REWRITE_REF_PATTERNS = re.compile(
        r"(dựa vào tóm tắt|theo tóm tắt|viết lại theo|xuất theo mẫu|"
        r"sửa theo tóm tắt|dựa trên tóm tắt|dựa vào nội dung.*tóm tắt|"
        r"theo nội dung.*đã tóm tắt|dựa vào kết quả|theo kết quả tóm tắt)",
        re.IGNORECASE | re.UNICODE,
    )
    text_for_pattern_check = raw_user_message if raw_user_message else query
    if _REWRITE_REF_PATTERNS.search(text_for_pattern_check):
        from llm_pipeline import tools as _tools_module
        saved_ctx = _tools_module.get_session_context(session_id, most_recent_only=True)
        if saved_ctx:
            injected_query = (
                f"{query}\n\n"
                f"[HỆ THỐNG - Context từ tóm tắt trước đó, dùng cho batch_rewrite_tool]:\n"
                f"{saved_ctx}"
            )
            print(f"[run_agent] 📌 Injected session context into query ({len(saved_ctx)} chars)")
        else:
            injected_query = query
    else:
        injected_query = query

    # Append the current (possibly context-injected) query
    messages.append({"role": "user", "content": injected_query})
    
    from pathlib import Path
    generated_files = []
    
    # Run the loop
    for step in range(max_steps):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tools=hf_tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Fast mode: skip chain-of-thought
            )
            
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            input_len = inputs.input_ids.shape[-1]
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    temperature=0.1,  # Low temp for accurate tool formatting
                    do_sample=True,
                )
            
            # Free input tensors immediately
            del inputs
            _safe_cuda_cleanup()
                
            response_text = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
            
            # Free generated tensors
            del generated_ids
            
        except (RuntimeError, torch.cuda.CudaError) as cuda_err:
            print(f"[Agent Error] CUDA error during generation: {cuda_err}")
            _safe_cuda_cleanup()
            messages.append({"role": "assistant", "content": "Lỗi GPU khi xử lý. Vui lòng thử lại."})
            break
        
        messages.append({"role": "assistant", "content": response_text})
        
        # Parse Qwen native <tool_call>...
        tool_call_match = re.search(r"<tool_call>\s*({.*?})\s*</tool_call>", response_text, re.DOTALL)
        if tool_call_match:
            try:
                tool_data = json.loads(tool_call_match.group(1))
                tool_name = tool_data.get("name")
                tool_args = tool_data.get("arguments", {})
                
                print(f"[Agent] ⚙️ Calling tool: {tool_name} with args: {tool_args}")
                
                if tool_name in tool_map:
                    # Invoke actual tool
                    tool_result = str(tool_map[tool_name].invoke(tool_args))
                    
                    # Track files (Revised, Exported, PDF)
                    file_patterns = [
                        r'[\w\-./\\, ]+_Revised\.docx',
                        r'[\w\-./\\, ]+_Export\.docx',
                        r'[\w\-./\\, ]+\.pdf',
                    ]
                    for pat in file_patterns:
                        m = re.search(pat, tool_result)
                        if m:
                            fname = Path(m.group()).name
                            if fname not in generated_files:
                                generated_files.append(fname)
                            
                    # Provide observation back to model
                    messages.append({
                        "role": "tool",
                        "name": tool_name,
                        "content": tool_result
                    })
                    continue # loop back to let LLM generate final answer based on observation
            except Exception as e:
                print(f"[Agent Error] Tool formatting error: {e}")
                messages.append({"role": "tool", "name": "error", "content": f"Lỗi gọi tool: {e}"})
                continue
        
        # ── FALLBACK: Model described a tool call in plain text without <tool_call> tag ──
        # Common with small models: "tôi cần gọi chat_tool" or "I'll use chat_tool"
        if step == 0:  # Only on first step to avoid infinite loops
            mentioned_tool = None
            response_lower = response_text.lower()
            # Check if model mentions a tool name in plain text
            for tname in tool_map:
                if tname in response_lower:
                    mentioned_tool = tname
                    break
            
            if mentioned_tool:
                print(f"[Agent] ⚠️ Model described '{mentioned_tool}' without calling it. Auto-invoking fallback...")
                # Determine the appropriate argument based on tool type
                actual_query = raw_user_message if raw_user_message else query
                if mentioned_tool == "chat_tool":
                    tool_args = {"query": actual_query}
                elif mentioned_tool == "compare_tool":
                    tool_args = {"input_text": actual_query}
                elif mentioned_tool == "edit_tool":
                    tool_args = {"instruction": actual_query}
                elif mentioned_tool == "batch_rewrite_tool":
                    # Context will be auto-fetched from session by the tool itself
                    tool_args = {"instruction": actual_query, "context": ""}
                elif mentioned_tool == "export_tool":
                    # Detect format from user message
                    fmt = "pdf" if "pdf" in actual_query.lower() else "docx"
                    tool_args = {"format": fmt}
                else:
                    tool_args = {"query": actual_query}
                
                try:
                    tool_result = str(tool_map[mentioned_tool].invoke(tool_args))
                    
                    # Track files (Revised, Exported, PDF)
                    for pat in [r'[\w\-./\\, ]+_Revised\.docx', r'[\w\-./\\, ]+_Export\.docx', r'[\w\-./\\, ]+\.pdf']:
                        m = re.search(pat, tool_result)
                        if m:
                            fname = Path(m.group()).name
                            if fname not in generated_files:
                                generated_files.append(fname)
                    
                    messages.append({
                        "role": "tool",
                        "name": mentioned_tool,
                        "content": tool_result
                    })
                    continue  # loop back to let LLM generate final answer
                except Exception as e:
                    print(f"[Agent Error] Fallback tool invocation error: {e}")
                
        # If no tool tag is found, this is the final answer
        break
        
    final_output = messages[-1]["content"]
    
    # Save the interaction to session history
    # Save the raw user message if provided to prevent history length explosion, otherwise save query
    user_msg_to_save = raw_user_message if raw_user_message else query
    _session_histories[session_id].append({"role": "user", "content": user_msg_to_save})
    _session_histories[session_id].append({"role": "assistant", "content": final_output})
    
    _safe_cuda_cleanup()
    
    return {
        "output": final_output,
        "files": generated_files
    }


def generate_raw(prompt: str, max_new_tokens: int = 4096) -> str:
    """
    Direct generation without agent (for structured outputs like JSON).
    Used by tools that need specific output formats.
    """
    model, tokenizer = load_model()
    
    messages = [
        {"role": "system", "content": "Bạn là trợ lý AI chuyên xử lý tài liệu. Luôn trả lời chính xác theo yêu cầu."},
        {"role": "user", "content": prompt},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Fast mode: skip chain-of-thought
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )
    
    generated_ids_trimmed = generated_ids[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)
    
    _safe_cuda_cleanup()
    return response.strip()


def generate_raw_batch(prompts: list[str], max_new_tokens: int = 4096) -> list[str]:
    """
    Direct batched generation (for parallel structure outputs like JSON across chunks).
    Speeds up repetitive document surgery immensely.
    """
    model, tokenizer = load_model()
    
    texts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "Bạn là trợ lý AI chuyên xử lý tài liệu. Luôn trả lời chính xác theo format JSON yêu cầu."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        texts.append(text)
        
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )
    
    responses = []
    for i in range(len(prompts)):
        generated_ids_trimmed = generated_ids[i][inputs.input_ids.shape[-1]:]
        resp = tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)
        responses.append(resp.strip())
        
    _safe_cuda_cleanup()
    return responses


def cleanup():
    """Free GPU memory."""
    global _model, _tokenizer
    
    if _model is not None:
        del _model
        _model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    
    _safe_cuda_cleanup()
    print("[LLM Engine] ✓ GPU memory cleaned up")