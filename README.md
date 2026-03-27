# PDF to DOCX Smart Converter

A complete pipeline to automatically analyze and convert PDF files (both Text-Based and Scanned) into fully formatted Microsoft Word Documents (DOCX).

## 🚀 Features

*   **Smart Auto-Detection**: Automatically classifies incoming PDFs as Text-Based, Scanned, or Hybrid.
*   **Fast Extraction**: Uses `PyMuPDF` for lightning-fast, high-accuracy text extraction on native text-based PDFs.
*   **Advanced OCR Pipeline**: Utilizes state-of-the-art AI for scanned document reconstruction:
    *   **DocLayout YOLOv10**: Detects complex layouts, bounding boxes, tables, and images accurately.
    *   **Qwen2.5-VL OCR**: Trình độ nhận diện chữ và format (in đậm, in nghiêng, danh sách, bảng biểu HTML) cao cấp từ ảnh scan. Nhỏ gọn và hỗ trợ tiếng Việt cực tốt.
*   **Layout Preservation**: Maintains the original reading order and relative positions of paragraphs using absolute positioning in DOCX (Textboxes).
*   **VRAM Optimized**: Supports 4-bit and 8-bit quantization (`bitsandbytes`) for Qwen2.5-VL on consumer GPUs.

## 🛠 Prerequisites

*   **OS:** Windows / Linux
*   **Python:** 3.10+
*   **GPU (Optional but recommended):** NVIDIA GPU (CUDA support) for running YOLO and OCR pipelines efficiently.

## 📥 Installation

1.  **Clone the repository or download the source code.**
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Make sure you have PyTorch installed with CUDA support if you intend to use the GPU. Check `requirements.txt` for the correct PyTorch installation command.*

3.  **Download Models**:
    *   The project expects the YOLO model weights (e.g., `doclayout_yolo_docstructbench_imgsz1024.pt`) to be in the root directory.
    *   **Qwen2.5-VL OCR Model**: The script uses `Qwen/Qwen2.5-VL-3B-Instruct` or a local version in `./Qwen2.5-VL-3B` by default. To download the current model locally, run:
        ```bash
        pip install -U "huggingface_hub[cli]"
        huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir Qwen2.5-VL-3B
        ```

## 💻 Usage

### 🌟 1. Automatic Processing (Recommended)
This script will analyze the PDF and automatically route it to the fastest/best conversion pipeline.

```bash
python src/pdf_processing/auto_process_pdf.py path/to/your/file.pdf
```

**Advanced Usage with OCR features for Scanned PDFs:**
```bash
python src/pdf_processing/auto_process_pdf.py path/to/your/file.pdf --output path/to/output.docx --enable_ocr --load_4bit
```
**Options:**
*   `--output`, `-o`: Output DOCX file path (Optional).
*   `--enable_ocr`: Force enable OCR processing (for scanned documents).
*   `--dpi`: DPI for converting PDF pages to images (Default: 300).
*   `--imgsz`: YOLO Image inference size (Default: 1024).
*   `--conf`: YOLO layout detection confidence threshold (Default: 0.1).
*   `--load_4bit`: Enable 4-bit quantization for the Qwen OCR model (Saves VRAM).
*   `--load_8bit`: Enable 8-bit quantization for the Qwen OCR model.

---

### 🔍 2. Check PDF Type Only
If you just want to analyze whether a PDF is native text or scanned without converting it.

```bash
python src/pdf_processing/check_pdf_type.py path/to/your/file.pdf
```

### 🧠 3. Test YOLO Layout Detection Only
To test bounding box detection and save annotated images to a folder, you can run the YOLO diagnostic script.

```bash
python src/pdf_processing/yolo_detect.py path/to/your/file.pdf --output-dir my_tests
```

## 🏗 Architecture & Code Structure

*   `src/pdf_processing/auto_process_pdf.py`: The main entry point. Decides which pipeline to run based on PDF characteristics.
*   `src/pdf_processing/check_pdf_type.py`: Script to analyze the text-to-page ratio of a PDF document.
*   `src/pdf_processing/processs_pdf_to_docs.py`: The Heavyweight Pipeline. Contains the logic for YOLO detection -> Qwen OCR Extraction -> Coordinate Mapping -> DOCX Reconstruction.
*   `src/pdf_processing/yolo_detect.py`: Diagnostics tool to visually preview the layout bounding boxes found by the YOLO model on your PDFs.

## 📝 Notes
*   **Text-Based Mode** focuses on speed and pure text extraction. It preserves text and basic reading order but drops complex layouts.
*   **Scanned (OCR) Mode** attempts to perfectly rebuild the document page-by-page. It handles multi-column layouts and tables but requires significantly more processing time and hardware.

---

## 🤖 LLM Document Intelligence Pipeline

A LangChain Agent powered by **Qwen3-4B** that can analyze, compare, and modify documents while preserving original formatting.

### Features
*   **Q&A**: Ask questions about uploaded documents (RAG with ChromaDB + BGE-M3)
*   **Compare**: Upload 2 documents and get a detailed comparison
*   **Edit**: Request content modifications — LLM generates changes, applies them to the original `.docx` preserving format

### Architecture
```
POST /chat → LangChain Agent (Qwen3-4B) → Auto-routes to:
  ├── chat_tool   → RAG search → answer / summarize
  ├── compare_tool → load 2 docs → diff
  └── edit_tool   → JSON modifications → doc surgery → revised .docx
```

### Setup

1.  **Install new dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Qwen3-4B model** (~8GB):
    ```bash
    huggingface-cli download Qwen/Qwen3-4B --local-dir Qwen3-4B
    ```

3.  **Start the API**:
    ```bash
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    ```
    API docs: http://localhost:8000/docs

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents/upload` | Upload Word/PDF |
| GET | `/documents/` | List documents |
| DELETE | `/documents/{doc_id}` | Delete document |
| POST | `/chat` | Unified chat (Q&A, compare, edit) |
| GET | `/download/{filename}` | Download revised files |
| GET | `/health` | Health check |

### Code Structure
*   `src/llm_pipeline/document_parser.py`: Parse Word/PDF → structured elements with IDs
*   `src/llm_pipeline/vector_store.py`: ChromaDB + BGE-M3 embedding management
*   `src/llm_pipeline/llm_engine.py`: Load Qwen3-4B + LangChain Agent
*   `src/llm_pipeline/tools.py`: 3 LangChain Tools (chat, compare, edit)
*   `src/llm_pipeline/doc_surgery.py`: Apply JSON modifications to .docx preserving format
*   `api/main.py`: FastAPI entry point
*   `api/routes/`: API route handlers
