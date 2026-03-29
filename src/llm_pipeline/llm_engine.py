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
    Native Qwen3 tool-calling loop.
    Bypasses LangGraph to ensure 100% accurate tool-call parsing with local HF models.
    """
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
        
    system_prompt = """Bạn là trợ lý xử lý tài liệu thông minh.
CHÚ Ý QUAN TRỌNG: KIẾN THỨC VÀ NỘI DUNG TÀI LIỆU KHÔNG ĐƯỢC CUNG CẤP TRONG PROMPT NÀY.
Bạn BẮT BUỘC phải gọi công cụ (tool) tương ứng để tìm thông tin trước khi trả lời.
- Dùng `chat_tool` để hỏi về nội dung, tóm tắt.
- Dùng `compare_tool` để so sánh 2 tài liệu.
- Dùng `edit_tool` để chỉnh sửa ĐIỂM NHỎ trong tài liệu.
- Dùng `batch_rewrite_tool` khi người dùng yêu cầu viết lại TOÀN BỘ tài liệu/file mẫu. Quan trọng: Trích xuất kết quả tổng hợp/thông tin từ lịch sử và TRUYỀN toàn bộ vào tham số `context` của tool này.

Luôn trả lời bằng tiếng Việt rõ ràng, ngắn gọn."""

    global _session_histories
    if session_id not in _session_histories:
        _session_histories[session_id] = []

    messages = [{"role": "system", "content": system_prompt}]
    
    # Append the last N history messages to prevent context overflow (e.g. max 10 turns)
    messages.extend(_session_histories[session_id][-20:])
    
    # Append the current augmented query
    messages.append({"role": "user", "content": query})
    
    from pathlib import Path
    generated_files = []
    
    # Run the loop
    for step in range(max_steps):
        text = tokenizer.apply_chat_template(
            messages,
            tools=hf_tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Fast mode: skip chain-of-thought
        )
        
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.1,  # Low temp for accurate tool formatting
                do_sample=True,
            )
            
        response_text = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
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
                    
                    # Track files
                    if "_Revised.docx" in tool_result:
                        m = re.search(r'[\w\-./\\]+_Revised\.docx', tool_result)
                        if m:
                            generated_files.append(Path(m.group()).name)
                            
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
                
        # If no tool tag is found, this is the final answer
        break
        
    final_output = messages[-1]["content"]
    
    # Save the interaction to session history
    # Save the raw user message if provided to prevent history length explosion, otherwise save query
    user_msg_to_save = raw_user_message if raw_user_message else query
    _session_histories[session_id].append({"role": "user", "content": user_msg_to_save})
    _session_histories[session_id].append({"role": "assistant", "content": final_output})
    
    torch.cuda.empty_cache()
    
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
    
    torch.cuda.empty_cache()
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
        
    torch.cuda.empty_cache()
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
    
    torch.cuda.empty_cache()
    print("[LLM Engine] ✓ GPU memory cleaned up")
