# CLAUDE.md — pdf_processing

Đọc file này đầu mỗi session, sau đó đọc `PROGRESS.md` để biết đang làm tới đâu.
Trả lời người dùng bằng tiếng Việt; code, identifier, commit message bằng tiếng Anh.

## 1. Mục tiêu và phạm vi

Chuyển **PDF (có chữ / scan / lai) và ảnh** sang **.docx giữ bố cục gần giống bản gốc**,
xử lý tốt **bảng, công thức toán, tiếng Việt**, **nhanh và chính xác**.
Thay pipeline cũ (DocLayout-YOLO + Qwen2.5-VL) bằng **MinerU + PyMuPDF**.

**Trọng tâm hiện tại: chỉ OCR / chuyển đổi.**
- Nhánh **`minerU`** chỉ chứa pipeline OCR: chatbot (`src/llm_pipeline/`, `api/`, `chatbot.*`, `Qwen3-4B/`) và pipeline cũ
  (`src/pdf_processing/`, YOLO + Qwen2.5-VL) đã xoá khỏi nhánh này (2026-11-24), vẫn còn nguyên trên `main`.
- Pipeline OCR mới là module độc lập, chạy bằng CLI. Việc nối vào chatbot để sau (mục 12).

## 2. Ràng buộc cứng

| Ràng buộc | Hệ quả thiết kế |
|---|---|
| Windows 11, 1 GPU 16 GB | MinerU dùng ONNX/DirectML hoặc LMDeploy/llama.cpp, **không có vLLM** |
| DOCX phải sửa được, bảng/công thức phải là đối tượng Word thật | **Không đặt chữ trong textbox.** Chữ, bảng, công thức ở dòng chảy của body. Textbox/anchor chỉ cho con dấu, chữ ký, logo |
| Tiếng Việt có dấu | Mọi tier phải qua kiểm tra tiếng Việt; MinerU docs **không nói** có hỗ trợ tiếng Việt → phải kiểm chứng |
| MinerU có thể xung đột `torch`/`transformers` với code cũ | MinerU chạy trong **venv/process riêng**, giao tiếp HTTP |
| License | MinerU ≥3.1: "MinerU Open Source License" (Apache 2.0 + điều kiện). `doclayout_yolo` đang dùng là AGPL → sẽ bỏ |

## 3. Flow tổng thể

```
File (PDF / jpg / png)
  │
  ├─[0] Cache: sha256(file)+page+tier ──trúng──► dùng lại IR
  │
  ├─[1] PageClassifier (PyMuPDF, ~ms/trang)   → nhãn cho TỪNG trang (mục 4)
  │
  ├─[1b] Tiền xử lý ảnh (chỉ SCAN/ảnh, `preprocess.py`): làm phẳng nền + deskew ±5° (đã làm); tìm khung giấy + nắn phối cảnh (chưa),
  │      **phóng to cạnh dài 3200 px** (bắt buộc — sửa chính tả hiệu quả nhất, mục 5.5)
  │
  ├─[2] Extract theo nhãn (song song CPU ‖ GPU)
  │     ├─ TEXT_SIMPLE   → PyMuPDF get_text("rawdict") + find_tables() (đã làm, gồm ô gộp)
  │     ├─ TEXT_COMPLEX  → render trang + MinerU standard (đang dùng; `flash` chưa nối)
  │     └─ SCAN / ảnh    → MinerU tier standard (tiếng Việt; basic chỉ cho ngôn ngữ khác)
  │
  ├─[3] Validate từng block ──hỏng──► OCR lại VÙNG đó ở tier khác, giữ block điểm cao nhất (mục 5)
  │
  ├─[4] Normalize → IR chung (Page / Block, mục 6) — giữ cả block `header`
  │
  ├─[4b] Sửa chính tả trên IR, chỉ chữ thuần (mục 5.5)
  │
  ├─[5] LayoutEngine + DocxRenderer (mục 7) ─► .docx
  │
  └─[6] Kiểm tra vòng kín: DOCX → PDF → so vị trí block với bản gốc; tràn trang → co chữ/khoảng cách, render lại (≤ 3 lần)
```

### 3.1 Bảng các bước

| # | Bước | Input | Output | Công cụ | Module (dự kiến) | Chạy trên |
|---|---|---|---|---|---|---|
| 0 | Cache lookup | file bytes | IR đã lưu hoặc miss | sha256 + JSON trên đĩa | `conversion/cache.py` | CPU |
| 1 | Phân loại trang | PDF | `list[PageKind]` | PyMuPDF | `conversion/page_classifier.py` | CPU |
| 2a | Extract trang có chữ đơn giản | trang PDF | IR blocks | PyMuPDF `dict`, `find_tables`, `get_images` | `conversion/pymupdf_extractor.py` | CPU |
| 2b | Extract trang có chữ phức tạp | trang PDF | middle_json | MinerU `flash` | `conversion/extractors/mineru_client.py` | CPU |
| 1b | Tiền xử lý ảnh | ảnh trang | ảnh đã nắn, cắt nền | OpenCV | `conversion/preprocess.py` | CPU |
| 2c | Extract trang scan / ảnh | trang PDF / ảnh | middle_json | MinerU `standard` (tiếng Việt) / `basic` (ngôn ngữ khác) | `conversion/extractors/mineru_client.py` | GPU |
| 3 | Validate + OCR lại vùng hỏng + ghép tier | IR blocks | IR blocks tốt nhất | luật kiểm tra (mục 5) + MinerU tier khác | `conversion/validators.py`, `conversion/router.py` | CPU + GPU |
| 4 | Normalize | middle_json / PyMuPDF dict | `Page`, `Block` | dataclass | `conversion/ir.py`, `conversion/normalize.py` | CPU |
| 4b | Sửa chính tả | IR | IR + log thay đổi | từ điển, bảng âm tiết, KenLM / LLM nhỏ | `conversion/spelling/` | CPU (+ GPU nếu LLM) |
| 5 | Layout + render DOCX | IR | .docx | python-docx + lxml, OpenCV, LaTeX→OMML | `conversion/layout.py`, `conversion/docx_renderer.py`, `conversion/tables.py`, `conversion/math_omml.py` | CPU |
| 6 | Kiểm tra bố cục vòng kín | .docx + ảnh gốc | .docx đã chỉnh | LibreOffice headless | `conversion/fit_check.py` | CPU |

Điểm vào: `conversion/pipeline.py::convert(path, out_path) -> Path`,
CLI: `python -m conversion <file> -o <out.docx>`.

## 4. Phân loại trang (bước 1)

### 4.1 Tín hiệu (PyMuPDF, mỗi trang)

| Tín hiệu | Cách lấy | Ý nghĩa |
|---|---|---|
| Ký tự hiển thị | `page.get_text("rawdict")`, bỏ span ẩn | Không có → nghi scan |
| Chữ ẩn | `page.get_texttrace()`, span `type == 3` | Lớp OCR ẩn của máy scan |
| Độ phủ ảnh | `page.get_image_info()` → diện tích hợp các bbox / diện tích trang | > ~85% → trang là một tấm ảnh |
| Chữ lỗi mã hoá | Tỉ lệ ký tự `U+FFFD`, Private Use `U+E000–F8FF`, ký tự điều khiển | Font thiếu ToUnicode |
| Chữ bị vẽ thành hình | Không có ký tự nhưng `len(page.get_drawings())` rất lớn | Text đã outline |
| Công thức | Tên font khớp `CMMI\|CMSY\|CMEX\|Cambria Math\|Symbol\|STIX`, hoặc ký tự `U+2200–22FF`, `U+1D400–1D7FF` | Cần MinerU để ra LaTeX |
| Bảng phức tạp | `page.find_tables()`: số ô giữa các hàng khác nhau (ô gộp) | Cần MinerU |

### 4.2 Quyết định (theo thứ tự)

```python
def classify_page(page) -> PageKind:
    chars = visible_chars(page)                       # bỏ span type==3
    img_cov = image_coverage(page)                    # 0..1
    bad = garbled_ratio(chars)                        # FFFD / PUA / control

    if is_image_file:                                 return SCAN
    if not chars and img_cov > 0.3:                   return SCAN
    if img_cov > 0.85 and len(chars) < 50:            return SCAN
    if has_hidden_text(page) and img_cov > 0.85:      return SCAN   # bỏ lớp OCR cũ
    if chars and bad > 0.2:                           return SCAN
    if not chars and many_drawings(page):             return SCAN   # text outline
    if has_math(page):                                return TEXT_COMPLEX  # bảng ô gộp: extractor tự xử lý
    return TEXT_SIMPLE                                # gồm cả trang trắng / bìa
```

Ngưỡng là giá trị khởi đầu, **chỉnh theo bộ mẫu** (mục 10).
Thay hàm cũ `analyze_pdf` (1 tỉ lệ cho cả file, `len(text) > 100`) bằng phân loại từng trang.

**Đã hiện thực (2026-11-24)**: `page_classifier.py` (luật trên) + `pymupdf_extractor.py` + `pipeline.py` theo từng trang:
- SCAN: render đúng độ phân giải ảnh gốc trong PDF (không bịa điểm ảnh) → `prepare` (phẳng nền, deskew, Lanczos 3200 px)
  → MinerU `standard`. Lớp OCR ẩn của máy scan (render mode 3) **bị bỏ**: thường sai dấu nặng.
- TEXT_COMPLEX (công thức): render vector ở 3200 px, không làm phẳng / deskew → MinerU `standard` → OMML.
- TEXT_SIMPLE: đọc thẳng chữ, font (Times/Arial), cỡ, đậm/nghiêng, màu, baseline; bảng kẻ `find_tables` → `w:tbl`
  (ô gộp suy từ bbox ô); ảnh → ảnh nổi; đường kẻ mảnh → vẽ lại. Dòng gộp thành đoạn chảy **chỉ khi** mỗi lần xuống
  dòng là do hết chỗ (từ đầu dòng sau không vừa dòng trước) và không phải khối căn giữa / căn phải; còn lại mỗi dòng
  đặt đúng vị trí. Toạ độ ngang lấy từ bbox **ký tự** (`rawdict`), không từ span (span tính cả dấu cách cuối).
- PDF chỉ có trang chữ → không khởi động MinerU / model sửa dấu.

## 5. Phát hiện lỗi MinerU và escalation (bước 3)

### 5.1 Tier

| Tier | Chi phí | Khi nào dùng |
|---|---|---|
| `flash` | ~0 (không model) | Trang **có lớp chữ** nhưng phức tạp. **Không dùng cho scan** (Flash OCR mất dấu) |
| `basic` | thấp | Scan **không phải tiếng Việt**; với tiếng Việt chỉ đáng tin phần layout/cấu trúc bảng, **không tin chữ** |
| `standard` | trung bình (VLM ~1.2B) | **Mặc định cho trang scan tiếng Việt** |
| `advanced` | cao | Block do `standard` có điểm thấp |

Số đo thật trên 3 mẫu (xem bảng đầy đủ trong `PROGRESS.md`): `basic` mất dấu (CER 7–12% trên scan);
**`standard` thắng `advanced` ở mọi mẫu**, cả trước lẫn sau phóng to → đường mặc định chỉ dùng `standard`;
`advanced` chỉ còn là phương án thử lại cho block điểm thấp, chưa chứng minh được lợi ích.

Nguyên tắc: **rẻ trước, chỉ nâng cấp phần hỏng**. Ưu tiên chạy lại **vùng** (crop theo bbox);
nếu MinerU xử lý crop kém thì fallback chạy lại theo trang (`page_range`).

### 5.2 Kiểm tra (mỗi block có điểm 0..1, lấy điểm của kiểm tra tệ nhất)

**A. Lỗi cứng:** HTTP lỗi, timeout, trang rỗng → retry 1 lần, rồi escalate.

**B. Sót nội dung (quan trọng nhất):** render trang 72 DPI, nhị phân hoá, đếm pixel mực **nằm ngoài
mọi bbox** MinerU trả về → nhiều = sót vùng, escalate đúng vùng đó. Block rỗng mà trong bbox có mực → lỗi.

**C. Chữ**

| Kiểm tra | Lỗi khi |
|---|---|
| Mật độ | Số ký tự thực ≪ ước lượng `(rộng / (0.5 × cỡ chữ)) × số dòng` (sót) hoặc ≫ (bịa) |
| Tiếng Việt | Tỉ lệ âm tiết hợp lệ (regex âm tiết / từ điển) < ~0.8 |
| Mất dấu | Văn bản tiếng Việt nhưng gần như toàn ASCII không dấu |
| Lặp | n-gram lặp nhiều lần (VLM kẹt vòng lặp) |
| Tin cậy | Dùng `score` của span nếu `middle_json` có (chưa xác nhận ở 4.x) |

**D. Bảng**

| Kiểm tra | Lỗi khi |
|---|---|
| Cấu trúc HTML | `lxml` parse lỗi; số cột thực (tính `colspan`/`rowspan`) lệch giữa các hàng |
| Đối chiếu ảnh | OpenCV morphology đếm đường kẻ ngang/dọc trong vùng; số hàng/cột lệch HTML |
| Nội dung | > ~40% ô rỗng; bảng chỉ 1 hàng hoặc 1 cột |

**E. Công thức:** `latex2mathml` lỗi; ngoặc `{}` không cân; rỗng; token lặp dài.

### 5.3 Chấm điểm

Mỗi kiểm tra đo một giá trị `x` rồi đổi ra điểm 0..1 bằng đường dốc tuyến tính:

```python
def ramp(x, good, bad):          # 1.0 tại good, 0.0 tại bad, tuyến tính ở giữa
    return max(0.0, min(1.0, (x - bad) / (good - bad)))

score  = 0.0 if any_gate_failed else min(soft_scores)   # min: không để điểm tốt che điểm hỏng
reason = tên kiểm tra có điểm thấp nhất                  # luôn ghi log
```

Ngưỡng khởi đầu (chỉnh lại bằng dữ liệu):

| Loại | Kiểm tra | `x` | good | bad |
|---|---|---|---|---|
| Chữ | Âm tiết tiếng Việt hợp lệ | tỉ lệ | ≥ 0.95 | ≤ 0.70 |
| Chữ | Ký tự lạ (`�`, PUA, control) | tỉ lệ | 0 | ≥ 0.05 |
| Chữ | Mật độ | thực / ước lượng | 0.6–1.6 | < 0.3 hoặc > 3 |
| Chữ | Lặp | max số lần lặp 3-gram | ≤ 2 | ≥ 6 |
| Chữ | Span score (nếu có) | trung bình | ≥ 0.9 | ≤ 0.6 |
| Bảng | HTML parse | **gate** | ok | lỗi → 0 |
| Bảng | Đồng đều cột | % hàng có số cột = mode | 1.0 | ≤ 0.7 |
| Bảng | Khớp đường kẻ (OpenCV) | độ lệch hàng/cột | 0 | ≥ 0.3 |
| Bảng | Ô rỗng | tỉ lệ | ≤ 0.15 | ≥ 0.5 |
| Bảng | Chữ trong ô | trung bình điểm chữ | như dòng "Chữ" | |
| Công thức | `latex2mathml` parse, ngoặc `{}` cân | **gate** | ok | lỗi → 0 |
| Công thức | Lặp token, độ dài LaTeX / bề rộng bbox | | | |
| Trang | Mực ngoài mọi bbox | tỉ lệ | ≤ 0.01 | ≥ 0.05 → tạo block giả điểm thấp |

### 5.4 Quyết định

| Điểm | Hành động |
|---|---|
| ≥ ngưỡng nhận (chữ **0.80**, bảng **0.85**, công thức **0.90**) | Nhận, dừng |
| Dưới ngưỡng, chưa tới `advanced`, còn ngân sách | Escalate **+1 tier** cho vùng đó |
| Dưới ngưỡng, đã `advanced` hoặc hết ngân sách | Giữ kết quả tốt nhất + chèn ảnh crop |

- Chỉ thay bằng kết quả tier mới khi điểm tăng **≥ 0.05**; hoà thì giữ tier rẻ hơn.
- Hết ngân sách: xếp hàng escalate theo `(1 − score) × diện tích`, sửa vùng hỏng nặng và lớn trước.
- Nhảy thẳng lên `standard` cho một loại vùng **chỉ khi** dữ liệu chứng minh `basic` gần như luôn hỏng loại đó.

```python
for b in blocks:
    tier = b.tier
    while score_block(b, img) < THRESHOLD[b.type] and tier < ADVANCED and budget.ok():
        tier = next_tier(tier)
        cand = mineru.parse_region(crop(img, b.bbox), tier)
        if score_block(cand, img) > score_block(b, img):
            b = cand                                  # chỉ thay khi tốt hơn thật
    if score_block(b, img) < THRESHOLD[b.type]:
        b.attach_fallback_image(crop(img, b.bbox))    # không mất thông tin, ghi log
```

- **Ngân sách**: tối đa X% vùng/file được escalate và tổng T giây; hết ngân sách → dùng ảnh crop.
- **Chỉnh ngưỡng bằng dữ liệu**: trên bộ mẫu có ground truth, chọn ngưỡng để bắt > ~90% block lỗi thật
  mà tỉ lệ escalate < ~10% vùng. Không đạt cả hai → cần thêm tín hiệu, không phải hạ ngưỡng.

### 5.5 Sửa chính tả (đã chốt sau thử nghiệm 2026-11-23)

**1. Phóng to ảnh trước OCR — sửa chính, bắt buộc.** Cạnh dài **3200 px** (Lanczos); PDF scan render ~400 DPI.
Đo thật, CER `standard` (gốc → 2400 px → 3200 px): bảng danh sách 2.17% → 0.46% → **0.15%**;
công văn 5.15% → **1.47%** → 1.61% (hết từ bịa, lấy lại quốc hiệu và "ĐỀ XUẤT"). 4000 px không tốt hơn.
~2.7 s/trang ở 3200 px (gốc ~1.4 s). **Không** có tác dụng với ảnh chụp form viết tay (mục 3 dưới).

**Phân tích lỗi còn sót** (3200 px `standard`, 2026-11-24): mẫu bảng còn **1** lỗi/133 từ (sai dấu);
mẫu công văn còn 16 chỗ/322 từ: **12 chỉ sai dấu** (tội, để, đón, nghỉ ×2, bố, k/gửi, Lưu, TRƯỞNG ×2, BCĐX),
4 còn lại: chữ ký "99" và ghi chú viết tay "Amy ch" bị OCR thành chữ → coi là ảnh (luật 1b);
"ngày04" (chữ dính số) và "duyệt/." (dấu câu) → **chấp nhận, không sửa**.

**1b. Chữ ký / viết tay bị OCR thành chữ → ảnh crop:** block chữ ngắn ở lề hoặc vùng chữ ký mà bị 5.6 gắn cờ
(S1 bất thường + S2 không ổn định) → chuyển thành ảnh crop. Không dùng regex / danh sách âm tiết.
**Không** tách / gộp chữ dính số dưới bất kỳ hình thức nào (quyết định 2026-11-24: rủi ro phá mã hiệu như
`A05`, `12h30`, `B5D13`, `K7.HND(05b)` lớn hơn lợi ích).

**2. Sửa dấu còn sót bằng model sửa chính tả tiếng Việt (seq2seq) có ràng buộc.** Lỗi còn lại đều là
**cùng chữ cái, sai dấu** ("tối phạm"→"tội phạm", "nghi"→"nghỉ", "Luu"→"Lưu", "TRƯỜNG KHOA"→"TRƯỞNG KHOA", "dón"→"đón").
- Model: **`protonx-models/protonx-legal-tc`** (T5 0.2B, chuyên văn bản pháp lý + hậu xử lý OCR; license v1.3-NC
  **phi thương mại**). Dùng thương mại → `nrl-ai/vn-spell-correction-base` (Apache 2.0). Đo 2026-11-24: công văn lỗi từ
  18 → **8** (ProtonX) / 10 (nrl-ai), ~3.6 s/trang gồm load, ~0.9 GiB VRAM. **Không dùng Qwen3-4B** (18 → 16, 23 s/trang).
- Tách câu ≤ 220 ký tự, chạy theo lô, beam 4; dòng toàn chữ hoa sửa ở chữ thường rồi viết hoa lại.
- Code kiểm tra từng từ: bỏ dấu phải **giống hệt** bản gốc **và giữ nguyên hoa/thường**; khác → bỏ thay đổi đó.
- Khoá: token có số / mã hiệu (`/ - & .` + chữ số, chữ hoa xen số); **tên riêng** (từ viết hoa không đứng đầu câu,
  ngoài dòng toàn chữ hoa).
- Không đụng LaTeX (inline thay `⟦F1⟧`), cấu trúc bảng (chỉ sửa chữ từng ô), ảnh.
- Mọi thay đổi → Track Changes trong DOCX. Đo CER trước/sau **và** số chữ đúng bị sửa sai (≈ 0).
- Đo bằng `scripts/bench_llm.py --model models/<tên> --middle <middle.json> --gt <gt.txt>`.

**3. Ảnh chụp điện thoại:** tìm khung giấy + nắn phối cảnh + cắt nền trước khi phóng to.
**Chữ viết tay** (MinerU trả `text_image`, hoặc block vẫn sai sau bước 2) → **giữ dạng ảnh crop** trong DOCX,
không chèn chữ sai.

Các lớp phức tạp hơn (ghép nhiều tier theo từng từ, từ điển cụm từ, KenLM, VLM đọc lại dòng) **tạm không làm**;
chỉ xem lại nếu số đo sau bước 1–2 còn lỗi đáng kể.

### 5.6 Phát hiện chữ đọc sai — tổng quát, không regex / không danh sách ngôn ngữ (chốt 2026-11-24)

Thay cho các kiểm tra theo luật ở 5.2-C (âm tiết hợp lệ, ký tự lạ, regex chữ Trung). Hai tín hiệu bổ sung nhau:

| Tín hiệu | Cách đo | Bắt được | Không bắt được |
|---|---|---|---|
| **S1. Độ hợp lý ngôn ngữ** | Mô hình ngôn ngữ tiếng Việt (chính LLM dùng ở 5.5) tính độ "bất ngờ" (surprisal) của từng từ trong ngữ cảnh — 1 lượt forward, không sinh chữ | Mọi thứ lạc ngữ cảnh: chữ Trung, "écrite", "Maryland", "sói gian", "Luu", "nghi học", "TRƯỜNG KHOA" — kể cả lỗi lặp lại ở mọi lần đọc | Lỗi vẫn hợp lý về nghĩa (tên người sai dấu) |
| **S2. Nhất quán thị giác** | Đọc lại **vùng bị S1 gắn cờ** theo cách khác (crop + phóng to khác / tier khác), so 2 lần đọc | Chữ bịa (mỗi lần đọc ra một kiểu) → phân biệt "ảnh khó đọc" với "chữ đúng nhưng hiếm" | Lỗi hệ thống (cả 2 lần đọc giống nhau) |

Số đo S2 (so 2 lần đọc độc lập, 2026-11-24):
- Đầu vào xấu (công văn ảnh gốc std vs adv; ảnh chụp đã cắt): bắt **73–87%** từ sai, precision 59–80%.
- Đầu vào tốt (3200 px): chỉ bắt **29%** — lỗi còn lại là sai dấu lặp lại ở mọi lần đọc → **S2 một mình không đủ, cần S1**.
- S1 **chưa đo** (cần trọng số LLM).

Thứ tự tối ưu (rẻ trước): S1 trên cả trang (~1 forward pass) → S2 **chỉ** cho vùng bị gắn cờ → hành động:

| S1 | S2 | Hành động |
|---|---|---|
| hợp lý | — | Giữ |
| bất thường | 2 lần đọc **khác nhau** | OCR lần 3 + bỏ phiếu; vẫn không ổn định → giữ **ảnh crop** vùng đó |
| bất thường | 2 lần đọc **giống nhau** | Sửa dấu có ràng buộc (5.5); không có phương án cùng bộ khung → **đánh dấu cho người duyệt** |

## 6. IR chung (bước 4)

Đơn vị: **point PDF (1/72 inch)**, gốc toạ độ **góc trên-trái** trang. Mọi nguồn phải quy về đúng đơn vị này.

```python
Page:  index, width_pt, height_pt, kind (PageKind), source ("pymupdf"|"flash"|"basic"|"standard"|"advanced"),
       blocks: list[Block]
Block: type, bbox (x0, y0, x1, y1), reading_order, tier, score,
       runs:  list[Run]        # paragraph/heading/list_item/caption
       level: int | None       # heading level / list level
       html:  str | None       # table (HTML từ MinerU hoặc dựng từ find_tables)
       latex: str | None       # equation_display
       image: Path | None      # image / stamp / fallback crop
Run:   text, bold, italic, underline, size_pt, font, color, latex_inline | None
Block.type ∈ {heading, paragraph, list_item, table, equation_display, image, caption,
              header, footer, page_number, stamp}
```

## 7. DocxRenderer (bước 5)

| IR | Word | Ghi chú |
|---|---|---|
| `Page` | Section: `pgSz` = kích thước trang thật, `pgMar` từ bbox nội dung | Mỗi trang PDF = 1 trang Word, ngắt trang giữa các trang |
| Nhiều cột | Section `w:cols` | Phát hiện bằng cụm bbox theo trục x |
| `heading` | Style `Heading N` | `level` từ MinerU `text_level` hoặc cỡ chữ PyMuPDF |
| `paragraph` | `w:p` | Căn lề theo bbox (tái dùng `determine_alignment_by_position`), indent = x0 − lề, spacing-before theo khoảng cách dọc |
| `Run` | `w:r` | Giữ đậm/nghiêng/gạch chân/cỡ/font/màu; font không có thì map về Times New Roman |
| `list_item` | Numbering Word | Không tự bịa list khi nguồn không có |
| `table` | `w:tbl` thật | Parse HTML (lxml) → `gridSpan` / `vMerge`; độ rộng cột theo bbox |
| `equation_display` | `m:oMathPara` | LaTeX → MathML (`latex2mathml`) → OMML (`MML2OMML.XSL` của Office, qua lxml XSLT); lỗi → ảnh crop |
| `latex_inline` | `m:oMath` trong đoạn | Như trên |
| `image` | Ảnh inline, kích thước theo bbox | |
| `header`/`footer` | Header/footer của section | `page_number` → field `PAGE`, không ghi số cứng |
| `stamp` (dấu, chữ ký, logo) | Ảnh anchor vị trí tuyệt đối | **Trường hợp duy nhất được dùng vị trí tuyệt đối** |

Không dựng toàn bộ tài liệu qua HTML → `htmldocx` (mất bbox, không hiểu LaTeX, ô gộp kém).
HTML chỉ dùng cho **nội dung bảng**. **Không dùng** Markdown/DOCX của MinerU: chúng bỏ block `header`
(mất quốc hiệu, tên cơ quan, số văn bản) và không có cỡ chữ / căn lề. Luôn render từ `middle.json`.

### 7.1 LayoutEngine — ĐÃ HIỆN THỰC (2026-11-24): `measure.py`, `layout.py`, `docx_renderer.py`, `fit_check.py`

MinerU không cho font / cỡ / đậm / màu → **mọi thứ đo trên ảnh** (ảnh đã phóng to mà OCR chạy trên đó).

| Thuộc tính | Cách làm đã chốt (đã thử trên Testcase) |
|---|---|
| Dòng chữ | Connected components: nét cỡ chữ cái tạo dòng (gom theo tâm y); nét nhỏ (dấu, chấm) không tạo dòng; nhóm thấp / ít nét gộp vào dòng gần nhất (so với chiều cao dòng chuẩn của trang). **Không dùng hình chiếu ngang** (dấu tiếng Việt làm dính dòng) |
| Độ rộng dòng | Mọi nét in đen trong dòng (kể cả gạch đầu dòng, dấu câu); **bỏ nét màu ở hai đầu** (nét ký, tick) — nét màu ở giữa (số viết tay) vẫn tính. Nét màu = lệch màu so với màu mực của trang ≥ 10 |
| Cỡ chữ | Độ rộng thật của glyph Times New Roman (Pillow) khớp với độ rộng đo: 1 dòng → chia trực tiếp; đoạn văn → mô phỏng ngắt dòng (có trừ thụt đầu dòng) tìm cỡ cho đúng số dòng + độ rộng dòng cuối, giới hạn theo khoảng cách dòng. Gom cỡ gần nhau (≤ 0.8 pt), thân bài dùng **một** cỡ; làm tròn 0.5 pt (Word lưu half-point) |
| Font | **Times New Roman** (thể thức văn bản hành chính). Phân biệt Arial bằng độ rộng **không tin được** ở độ phân giải scan → bỏ |
| Đậm | **Độ đen lõi nét** (trung bình 1/3 điểm ảnh tối nhất) < 0.83 × trung vị trang. Đo: đậm 18–45, thường ≥ 47. Độ dày nét / tỉ lệ phủ mực **không phân biệt được** (đã thử) |
| Nghiêng | Tỉ lệ cột trống sau khi xiên 12° > 1.1 (nghiêng 1.21–1.31, đứng ≤ 0.95) — mức block |
| Màu / chữ viết tay | **Lệch màu tương đối** với màu mực của chính trang (`ink_reference`, bền với ảnh ám màu): trung vị ≥ 16 (in ≤ 7.4, in dưới con dấu 13.9, bút/dấu 22–59). Nét bút ≤ 2 ký tự cỡ > 1.8× cỡ thường cũng là ảnh. Crop ảnh nổi: mực in trong vùng dấu được làm trong suốt → **ảnh nổi** PNG nền trong suốt đúng vị trí (chữ ký, "Đồng ý", nét ký). MinerU `image` (con dấu…) cũng vậy |
| Khối song song | **Recursive XY-cut** (băng ngang → cột) → bảng ẩn không viền, độ rộng cột theo x. Nhóm hàng theo Y kiểu code cũ đã thử và **sai**. Một dòng ảnh chứa 2 cột (quốc hiệu ‖ cơ quan) → tách tại khe **rộng nhất ở vùng giữa trang** (30–70%), ≥ 3× khe trung vị và ≥ 1 chiều cao chữ không dấu, 2 mảnh ở 2 nửa trang |
| Dòng MinerU ↔ dòng ảnh | Quy hoạch động (`_match_lines`): 1 dòng `\n` trải k dòng ảnh (đoạn chảy) **hoặc** nhiều dòng `\n` trên 1 dòng ảnh (lưới bài tập "a) … b) …": **tab stop đúng khe trắng đo được**; không có khe → nối bằng dấu cách) |
| Công thức | MinerU `equation_inline` / `interline_equation` (LaTeX) → MathML (`latex2mathml`) → **OMML qua `MML2OMML.XSL` của Office** = công thức Word sửa được (`math_omml.py`). Độ rộng ước lượng theo cấu trúc (phân số = phần rộng hơn, mũ nhỏ hơn) bằng Cambria Math. Không chuyển được → chữ hiển thị. Sửa chính tả **không** chạm LaTeX. Dòng có công thức: nhóm chồng dọc > 15 % gộp làm 1 dòng (phân số cao); dòng chữ thường giữ 40 % |
| Đậm trên ảnh rất nét | Mọi nét đều đen (trung vị độ đen < 25) → độ đen không phân biệt được → đậm theo loại block MinerU (`title`, `paragraph_title`, `doc_title`). Chỉ kết luận đậm theo độ đen khi đậm hơn trung vị ≥ 8 mức xám |
| Đoạn văn chảy | Block MinerU `text`/`list` nhiều dòng, ≥ 4 từ/dòng, không phải khối ngắn căn giữa (bền với ảnh cong); justify nếu các dòng đều đầy; thụt đầu dòng chung cho cả trang |
| Dòng độc lập | Tiêu đề, quốc hiệu, chữ ký…: mỗi dòng 1 paragraph, **căn giữa trong khung mực của chính dòng đó** (sai số độ rộng font chia đều 2 bên); khối nhiều dòng không có `\n` → chia chữ theo tỉ lệ độ rộng dòng đo được |
| Vị trí dọc | Line spacing **exact** = khoảng cách dòng đo; space-before = khoảng trống đo được |
| Bảng | `w:tbl` thật, ô gộp; cột từ đường kẻ dọc (ngưỡng **thích nghi** — đường kẻ mờ sau phóng to); thiếu đường ngang → chia đều chiều cao bảng; cỡ chữ **đo trong từng ô** (trung vị có trọng số theo độ dài chữ, tiêu đề tách riêng); căn lề từng ô đo được, cột theo đa số |
| Gạch trang trí | Đường ngang ngắn ngay dưới một dòng in (tên cơ quan, tiêu ngữ, trích yếu) → vẽ lại sạch từ điểm ảnh của đường kẻ (không crop: dính chân chữ) |
| **Vòng kín** (`fit_check.py`) | Word xuất PDF (PowerShell COM, không cần cài thêm) → baseline mỗi dòng (PyMuPDF) + tâm mực theo glyph → dịch từng đoạn đúng phần lệch; giữ lần tốt nhất. Mẫu scan: lệch dọc TB 1–3.7 pt → **0.0 pt**. Tràn sang trang thừa → phạt `OVERFLOW_PENALTY`, co khoảng dòng 4–15% riêng tài liệu đó (không dưới cỡ chữ), giữ lần tốt nhất. Dòng độc lập không bao giờ ngắt (rộng = chữ thật × 1.04) |

Bổ sung 2026-11-24 (PDF): cỡ đoạn văn khớp **độ rộng mọi dòng** (không chỉ dòng cuối); khối căn giữa (lề trái lệch, tâm thẳng)
không gộp đoạn; XY-cut dùng khung mực đo được (`Para.ink`), không dùng khung đã nới; chia chữ cho các mảnh dòng bằng quy
hoạch động giữ cỡ suy ra đồng đều; crop con dấu chỉ xoá nét đen nằm trong dòng chữ đã dựng lại (giữ chữ ký đen).

**Quy tắc chung (2026-11-24):** mọi ngưỡng bố cục phải **tương đối theo chính trang** (khe trung vị, màu mực trang, chiều cao chữ không dấu, loại block MinerU), không theo số tuyệt đối của 1 mẫu; sửa xong chạy lại **cả** `Testcase`.

Chưa làm: kiểu chữ **trong** đoạn (cụm nghiêng "(có danh sách kèm theo)", số "1." đậm); ngắt dòng từng chữ trong đoạn
có thể lệch 1 từ so với bản gốc; ảnh chụp điện thoại (cần bước 1b tìm khung giấy + nắn phối cảnh trước).

## 8. Hiệu năng

- MinerU chạy **server thường trực** (model load 1 lần): `mineru-kit api-server` (small models) và `mineru-kit vlm-server` (VLM).
  Port: MinerU API = **8001**, VLM server = **30000** (tránh 8000 của FastAPI cũ).
- Gộp các trang cùng tier vào 1 request (`page_range="2,5-8"`), không gửi từng trang.
- CPU (PyMuPDF, render DOCX) chạy song song GPU (OCR): `asyncio` + `httpx`, trang nào xong render trang đó.
- Không tự render ảnh 300 DPI như code cũ; để MinerU tự lo (chỉ render 72 DPI cho kiểm tra mục 5.2-B).
- Cache theo `sha256(file) + page + tier`.
- Tốc độ (giây/trang theo từng tier) **chưa đo** — lấy số từ spike rồi ghi vào `PROGRESS.md`.

## 9. Code cũ: bỏ / giữ

| Bỏ (sau khi pipeline mới đạt benchmark) | Giữ / tái dùng |
|---|---|
| `src/pdf_processing/yolo_detect.py` | `determine_alignment_by_position` (`processs_pdf_to_docs.py:35`) |
| YOLO + Qwen2.5-VL OCR, lọc trùng, sort reading order, lọc output rác trong `processs_pdf_to_docs.py` | Helper font / bảng không viền trong Step 8 của `processs_pdf_to_docs.py` |
| Textbox anchor cho chữ (Step 6) | `HtmlToDocx` làm fallback cho bảng |
| `analyze_pdf` dạng tỉ lệ cả file (`auto_process_pdf.py`) | |
| Dependency: `doclayout_yolo`, `ultralytics`, `qwen-vl-utils`, `pdf2docx`, `pdf2image`; thư mục `Qwen2.5-VL-3B` | |

**Đã xoá trên nhánh `minerU` (2026-11-24)** theo yêu cầu người dùng; bản cũ còn trên `main`. Các hàm "giữ / tái dùng"
ở trên đã được viết lại trong `conversion/` (không import code cũ).

## 10. Kiểm chứng chất lượng

Bộ mẫu 10–20 file thật trong `samples/` (gitignored): công văn scan tiếng Việt, bảng có ô gộp, công thức,
văn bản 2 cột, PDF có chữ, PDF lai, ảnh chụp, scan có lớp OCR ẩn.

| Chỉ số | Cách đo |
|---|---|
| Phân loại trang | Độ chính xác nhãn so với nhãn gán tay |
| Chữ | CER so với ground truth gõ tay |
| Bảng | TEDS (so cấu trúc HTML) |
| Công thức | Tỉ lệ LaTeX chuyển OMML thành công + kiểm tra mắt |
| Bố cục | DOCX → PDF (LibreOffice headless) → SSIM với ảnh trang gốc |
| Escalation | % vùng bị escalate, % lỗi thật bị bắt |
| Tốc độ | giây/trang theo từng tier, tổng thời gian/file |

Không báo "xong" nếu chưa chạy benchmark và ghi số vào `PROGRESS.md`.

## 11. Lệnh chạy

- **Flow hoàn chỉnh, 1 lệnh, chạy theo lô** (`src/conversion/pipeline.py`) — **chỉ ghi file .docx**:
  `python scripts\convert.py Testcase -o D:/ket_qua` (mọi ảnh **và PDF** → `D:/ket_qua/<stem>.docx`);
  `python scripts\convert.py scan.pdf -o D:/ket_qua/cong_van.docx` (1 file → đúng đường dẫn đó).
  PDF: mỗi trang phân loại riêng (mục 4), 1 file DOCX / input, 1 trang Word / trang PDF.
  Phóng to 3200 px → MinerU `standard` (`scripts/mineru_parse.py`, venv riêng, chỉ ra middle.json) song song với load
  model sửa dấu → sửa dấu cả lô 1 lượt GPU → bố cục + công thức OMML → vòng kín (mỗi vòng **1 lần mở Word cho cả lô**).
  File trung gian ở thư mục tạm, tự xoá. `--passes 2` (mặc định, chính xác nhất) / `1` / `0` (không cần Word).
  `--keep-work <dir>` giữ trung gian + `debug/<stem>.compare.png`. Có `<stem>.gt.txt` cạnh ảnh → báo CER.
  Cần: `.venv-mineru/`, `models/protonx-legal-tc/`, Microsoft Word (khi `--passes` > 0).
- Chỉ dựng lại bố cục từ work đã giữ: `python scripts/render_layout.py <work> [--stems <tiền tố>]`
- Test: `python -m pytest tests -q`
- Spike MinerU (trong venv riêng): `.venv-mineru\Scripts\python scripts\spike_mineru.py Testcase --tiers basic standard advanced --warmup`
  → `Testcase/output/<stem>/<tier>/{result.md, middle.json, content_list.json, mineru.docx}` + `timings.json`
- Chấm CER: `python scripts\eval_ocr.py Testcase` (cần `Testcase/<stem>.gt.txt` gõ tay)
- Trên Windows đặt `PYTHONUTF8=1` để tránh lỗi in tiếng Việt / tiếng Trung ra console.
- `Testcase/` chứa dữ liệu thật (tên người) — ảnh, `*.gt.txt`, `output/` đều bị gitignore, **không commit**.
- Cài đặt: `requirements.txt` (venv chính), `requirements-mineru.txt` (`.venv-mineru`), model
  `huggingface-cli download protonx-models/protonx-legal-tc --local-dir models/protonx-legal-tc` (weights bị gitignore). Xem `README.md`.

## 12. Để sau (ngoài phạm vi hiện tại)

Nối pipeline mới vào chatbot: `document_parser.convert_pdf_to_docx` gọi `conversion.pipeline.convert`.
Lưu ý khi làm: `parse_docx` / `doc_surgery` bỏ qua `<w:txbxContent>` (lý do thêm cho quy tắc không dùng textbox);
Qwen3-4B 4-bit chiếm ~10.4/16 GiB VRAM nên VLM của MinerU phải đi qua hàng đợi.

## 13. Rủi ro / câu hỏi mở

Đã trả lời (spike 2026-11-23, mới 1 mẫu — cần thêm mẫu để chắc):
1. ~~Tiếng Việt?~~ `flash`/`basic` (PP-OCR) **mất dấu** ("Nguyn Vn An" thay cho "Nguyễn Văn An"); `standard`/`advanced` (VLM) đọc được,
   còn lỗi dấu thanh ("ủy"↔"úy", "VỀT"↔"VẾT", "Trường"↔"Trưởng"). Cân nhắc bước sửa chính tả tiếng Việt sau OCR.
4. ~~DOCX gốc MinerU?~~ Có bảng Word thật (`Table Grid`), không textbox, nhưng **không giữ bố cục**:
   trang mặc định A4 lề 2 cm, không căn giữa, không cỡ chữ → vẫn cần `DocxRenderer` riêng (hoặc hậu xử lý DOCX của MinerU).

5. ~~Field `middle_json`?~~ `bbox` chuẩn hoá **0..1**, `index` = thứ tự đọc, type `header`/`doc_title`/`paragraph_title`/
   `text`/`list`/`table`/`image`; **không có** cỡ chữ, font, đậm, căn lề, score (chi tiết trong `PROGRESS.md`).
   **Markdown và DOCX của MinerU bỏ block `header`** (mất quốc hiệu, số văn bản) → renderer phải đọc `middle.json`.

Còn mở:
2. `basic` xử lý bảng/công thức tới mức nào? (bảng ô gộp: tier VLM ra đúng `rowspan`; công thức: chưa có mẫu)
3. MinerU nhận ảnh crop 1 vùng có ổn không (escalate theo vùng)?
6. Tốc độ trên PDF nhiều trang (mới đo ảnh đơn).
7. Ảnh chụp điện thoại (nghiêng, nền lẫn giấy khác): cần tiền xử lý tìm khung + nắn phối cảnh trước MinerU.

Môi trường đã chạy: `mineru` **4.2.3** trong `.venv-mineru/` (gitignored), ONNX DirectML GPU + VLM llama-cpp,
`mineru-kit hardware detect` khuyến nghị đủ 4 tier. API Python: `mineru.parser.parse(path, tier=...)`,
`mineru.render.api.render(result.middle_json, RenderFormat.DOCX)`.
