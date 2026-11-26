# PROGRESS

## Goal
Thay pipeline OCR cũ (DocLayout-YOLO + Qwen2.5-VL) bằng MinerU + PyMuPDF, xuất .docx giữ bố cục,
xử lý tốt bảng / công thức / tiếng Việt, nhanh. **Chỉ tập trung OCR; chatbot để sau.**
Thiết kế đầy đủ: `CLAUDE.md`.

## Done
- 2026-11-22: Phân tích pipeline cũ, tra cứu MinerU 4.x, chốt thiết kế (phân loại từng trang,
  escalation tier rẻ → đắt theo vùng, IR chung, một DocxRenderer, không dùng textbox cho chữ).
- 2026-11-23: Ghi thiết kế vào `CLAUDE.md`; thu hẹp phạm vi về OCR; thêm chi tiết phân loại trang,
  phát hiện lỗi và chấm điểm escalation (mục 4, 5).
- 2026-11-23: **Spike MinerU 4.2.3** (`.venv-mineru/`, `scripts/spike_mineru.py`, `scripts/eval_ocr.py`)
  trên 1 ảnh: bảng danh sách tiếng Việt 10 người (`Testcase/9d6a89f4-….jpg`).

### Kết quả spike (RTX 5070 Ti, ONNX DirectML + VLM llama-cpp)

| Tier | CER | Thời gian (warm) | Ghi chú |
|---|---|---|---|
| `flash` | 7.4% | — (lần đầu 21.7 s gồm tải model) | Mất dấu; bảng ra dạng text, không phải bảng |
| `basic` | 7.4% | 0.39 s | Bảng đúng cấu trúc 11×4, nhưng mất dấu |
| `standard` | **2.2%** | 0.96 s | Dấu gần đúng hết; lỗi dấu thanh: "tả", "ủy", "VỀT TỔI", "Trương đoán", "Än" |
| `advanced` | 7.6% | 1.50 s | Dấu tốt nhất ("tá", "đoàn") nhưng **làm rơi dòng cuối** → CER cao |

DOCX gốc MinerU: bảng Word thật, không textbox, nhưng A4 mặc định, không căn lề / cỡ chữ → không giữ bố cục.
Lần chạy đầu tải model từ Hugging Face (~vài chục giây).

### Spike lần 2 (2026-11-23, thêm 2 mẫu, chưa có ground truth → chưa có CER)

| Mẫu | Tier | Thời gian (warm) | Nhận xét |
|---|---|---|---|
| `aec455ba` công văn đề xuất (scan phẳng, 2 cột đầu trang, bảng ô gộp, chữ ký) | `basic` | 0.54 s | — |
| | `standard` | 1.37 s | Thân bài khá; khối đầu trang sai dấu nặng ("CÔNG HÒA XĂ HÒI CHU NGHÍA VIÊT NAM"); lẫn từ lạ ("écrite", "donc", "sói gian") |
| | `advanced` | 2.81 s | Khối đầu trang **đúng**; thân bài **tệ hơn** ("cubic", "佛山", "Across Ektron", "Maryland"); chữ ký/chữ viết tay → ảnh `text_image` |
| `e57c5aec` ảnh chụp điện thoại giấy giới thiệu (nghiêng, trong túi nhựa, có giấy khác ở nền) | `basic` | 3.62 s | Mất dấu, chữ dính nhau |
| | `standard` | 1.40 s | Chữ in đọc được; chữ viết tay sai nhiều; lẫn chữ của giấy ở nền ("Cuom") |
| | `advanced` | 2.29 s | — |

Phát hiện quan trọng:
- **Markdown và DOCX gốc của MinerU bỏ hẳn block `header`** → mất quốc hiệu, tên cơ quan, số văn bản, ngày tháng.
  Dữ liệu vẫn có trong `middle.json` → renderer riêng phải đọc `middle.json` và giữ `header`.
- Dòng tiêu đề "ĐỀ XUẤT" bị mất ở cả 2 tier VLM (kiểm tra B "mực ngoài bbox" phải bắt được).
- Bảng ô gộp dọc: cả `standard` và `advanced` đều ra đúng `rowspan="2"`.
- **Không tier nào thắng trên mọi block**: header tốt nhất ở `advanced`, thân bài tốt nhất ở `standard`
  → xác nhận cần chấm điểm và ghép block tốt nhất giữa các tier.
- Ảnh chụp điện thoại cần bước tiền xử lý: tìm khung giấy + nắn phối cảnh, cắt bỏ nền.

### Spike lần 3: phóng to ảnh trước OCR (cạnh dài 2400 px, Lanczos)

| Mẫu | Tier | CER trước → sau | Thời gian | Nhận xét |
|---|---|---|---|---|
| `9d6a89f4` bảng danh sách | `standard` | 2.17% → **0.46%** | 0.96 → 1.95 s | Còn "THẠM", "só" |
| | `advanced` | 7.59% → 6.50% | 2.49 s | Vẫn rơi dòng cuối |
| `aec455ba` công văn | `standard` | (chưa có GT) | 1.37 → 1.99 s | Quốc hiệu, tên cơ quan **đúng**; lấy lại "ĐỀ XUẤT"; hết "écrite/donc/sói gian". Còn sai dấu: "tối/tôi phạm", "nghi", "k/gùi", "Luu", "TRƯỜNG KHOA", "dón", "đề đưa", "BCDX" |
| `e57c5aec` ảnh chụp | `standard` | (chưa có GT) | 1.95 s | Vẫn tệ: nền lẫn chữ, chữ viết tay sai → cần cắt khung giấy; viết tay giữ dạng ảnh |

→ Phóng to là cách sửa chính tả hiệu quả nhất đã đo được. Lỗi còn lại chủ yếu là sai dấu cùng chữ cái.

Thử thêm mức phóng to (`standard`): mẫu bảng CER 2400 px 0.46% → **3200 px 0.15%** → 4000 px 0.46%; ~2.7 s/trang.
Mẫu công văn ở 3200 px: thêm đúng "k/gửi", 1/3 "tội phạm"; vẫn sai "tôi phạm" ×2, "nghi" ×2, "Luu", "TRƯỜNG KHOA" ×2,
"dón", "đề đưa" → độ phân giải không sửa thêm được, cần bước sửa dấu trên chữ. **Chốt 3200 px.**

### Đo đủ 3 mẫu (2026-11-24) — `scripts/eval_ocr.py` chấm từ `middle.json` (gồm `header`, ô bảng)

Ground truth gõ tay cho cả 3 mẫu (`Testcase/*.gt.txt`, gitignored). CER = lỗi ký tự (tính cả thứ tự đọc),
WErr = lỗi từ không phụ thuộc thứ tự. Mẫu 3 GT gồm cả chữ viết tay điền vào form.

| Mẫu | Ảnh gốc `standard` | 2400 px `standard` | 3200 px `standard` | 3200 px `advanced` |
|---|---|---|---|---|
| 1. Bảng danh sách (scan) | 2.17% / 10.5% | 0.46% / 2.3% | **0.15% / 0.8%** | 0.62% / 1.9% |
| 2. Công văn (scan 565 px) | 5.15% / 13.7% | **1.47% / 4.8%** | 1.61% / 5.0% | 3.28% / 5.1% |
| 3. Ảnh chụp điện thoại (form viết tay) | 59.3% / 51.4% | 53.6% / 44.0% | 53.1% / 42.5% | 86.6% / 88.2% |
| 3. — cắt tay vùng giấy + 3200 px | | | **28.4%** / 45.5% | 48.2% / 47.1% |

Kết luận:
- Scan: phóng to giảm CER **3.5–14 lần**; 2400 và 3200 px gần như ngang nhau (chọn 3200: tổng tốt hơn chút).
- `standard` thắng `advanced` ở **mọi** mẫu → bỏ `advanced` khỏi đường mặc định.
- **Đính chính**: `advanced` không "rơi dòng" — dòng cuối có trong `middle.json`, chính markdown của MinerU bỏ nó.
- Ảnh chụp form viết tay: phóng to gần như vô ích; cắt vùng giấy giúp CER 53% → 28% nhưng vẫn không dùng được.
  Lỗi chính: chữ viết tay + vân túi nhựa làm sai dấu chữ in → cần nắn phối cảnh + **giữ chữ viết tay dạng ảnh**.

### Thử LLM sửa dấu: Qwen3-4B BF16 (2026-11-24, `scripts/bench_llm.py`)

Trọng số tải về `Qwen3-4B/` (7.6 GB, gitignored). torch 2.8 + CUDA 12.8, transformers 4.56, RTX 5070 Ti.
- Load 4–6 s, VRAM 7.5 GiB (đỉnh 8.3 GiB). Sinh chữ **~28 token/s** (transformers, sdpa).
- Yêu cầu trả danh sách "sai -> đúng": **model bỏ qua**, viết lại cả văn bản → bỏ cách này.
- Viết lại từng đoạn + code chỉ nhận đổi dấu cùng bộ khung: mẫu công văn (3200 px) lỗi từ **18 → 17**/322,
  chỉ sửa được "dón→đón", không làm hỏng chữ đúng; **22 s/trang**.
- S1 (surprisal 1 lượt, 0.45–0.8 s/trang): gắn cờ đúng "Luu:", "dón", "TRƯỜNG" ×2 nhưng cũng gắn cờ chữ đúng
  ("HỌC", "đốc;", "Căn,") và sót "tôi", "nghi", "đề", "bổ".
→ **Qwen3-4B không đủ tốt và quá chậm cho bước sửa dấu.** Cần thử model khác bằng cùng script.

Thử lại prompt (cùng dữ liệu, `--prompt` / `--doc` / `--think`; ví dụ mẫu tự viết, không lấy từ Testcase):

| Prompt | Lỗi từ (trước 18) | Sửa được | Thời gian |
|---|---|---|---|
| plain (chép nguyên văn) | 17 | dón | 22 s |
| fewshot (nói rõ "thường có lỗi dấu" + 2 ví dụ) | 16 | dón, Luu | 23 s |
| fewshot + cả tài liệu 1 lần | 16 | tôi, dón | 23 s |
| fewshot + cả tài liệu + thinking | 16 | tôi ×2 | 96 s |

→ Prompt có ảnh hưởng nhưng nhỏ (+1 lỗi được sửa); thinking chậm gấp 4 mà không tốt hơn.
**Giới hạn chính là năng lực model 4B**, không phải prompt. Không có biến thể nào làm hỏng chữ đúng.
Mặc định script: `--prompt fewshot`.

### Thử model sửa chính tả chuyên tiếng Việt (2026-11-24, `scripts/bench_llm.py`, tự nhận diện seq2seq)

Tải về `models/` (gitignored). Cài thêm `sentencepiece` vào môi trường chính (tokenizer BARTpho cần).
Cách chạy: tách câu ≤ 220 ký tự, lô 16, beam 4, fp16; dòng toàn chữ hoa → sửa ở chữ thường rồi viết hoa lại;
code chỉ nhận đổi dấu **cùng bộ khung, giữ nguyên hoa/thường**, khoá token có số hoặc `/ & ( )`.

| Model | Kích thước / license | Công văn: lỗi từ (trước 18) | Bảng (trước 1) | Thời gian (gồm load) | VRAM |
|---|---|---|---|---|---|
| **protonx-models/protonx-legal-tc** (T5 0.2B) | 862 MB / **v1.3-NC: phi thương mại** | **8** — sửa đủ 10 lỗi dấu | 1 | 3.6 s | 0.9 GiB |
| nrl-ai/vn-spell-correction-base (ViT5) | 862 MB / Apache 2.0 | 10 — sót "TRƯỜNG" ×2 | 1 | 3.6 s | 0.9 GiB |
| bmd1905/vietnamese-correction-v2 (BARTpho) | 1.5 GB / Apache 2.0 | 13 | 1 | 3.0 s | 1.1 GiB |
| Qwen3-4B (để so sánh) | 7.6 GB / Apache 2.0 | 16 | — | 23 s | 7.6 GiB |

- Không model nào làm hỏng chữ đúng (sau khi bắt buộc giữ hoa/thường: nrl-ai từng đổi "Tham"→"tham").
- 8 lỗi còn lại của công văn đều nằm ngoài phạm vi sửa dấu: "k/gửi", "BCĐX" (token bị khoá), "ngày04" (chữ dính số,
  đã quyết định không sửa), "duyệt/." (dấu câu), "Amy ch", "99" (viết tay / chữ ký → luật 1b chuyển thành ảnh).
- Lỗi duy nhất của mẫu bảng ("Trường đoàn" trong 1 ô bảng 2 từ) không model nào sửa: quá ít ngữ cảnh.
→ **Chọn ProtonX** nếu dự án phi thương mại; **nrl-ai base** nếu cần dùng thương mại (kém 2 lỗi trên mẫu này).

### Pipeline chính tả chạy thật trên toàn bộ Testcase (2026-11-24)

Người dùng xác nhận: dự án **nội bộ, phi thương mại** → dùng ProtonX.
Lệnh: `python scripts/run_pipeline.py Testcase -o Testcase/output`
(phóng to 3200 px → MinerU `standard` → `src/conversion/spelling.py` sửa dấu trong `middle.json` → render → chấm).
Kết quả: `Testcase/output/<stem>/standard/` (OCR gốc) và `standard_fixed/` (`middle.json`, `changes.json`, `result.md`, `mineru.docx`).

| Mẫu | Trước sửa CER / WErr | Sau sửa CER / WErr | Số thay đổi |
|---|---|---|---|
| Bảng danh sách | 0.15% / 0.75% | 0.15% / 0.75% | 0 |
| Công văn | 1.61% / 4.95% | **0.94% / 1.86%** | 10, **đều đúng** |
| Ảnh chụp form tay | 53.1% / 42.5% | 51.5% / 37.4% | 13: 8 đúng, 1 **sai** ("lưu trú"→"lưu trữ", chữ của tờ giấy ở nền), 4 không rõ |

Thời gian (3 ảnh): phóng to 0.2 s; MinerU ~2.5 s/trang (+ load/warm-up); load ProtonX 6.3 s; sửa dấu 0.8–1.9 s/trang.
`spike_mineru.py` giờ lưu `middle.json` **kèm `image_base64`** (cần để render lại chữ ký / con dấu sau khi sửa).
Còn lại trên công văn: "k/gửi", "BCĐX" (token khoá), "ngày04", "duyệt/.", "Amy ch", "99" (viết tay/chữ ký chưa chuyển thành ảnh).

### Dựng DOCX giữ bố cục (2026-11-24) — thiết kế chốt ở CLAUDE.md mục 7.1

Code: `src/conversion/{measure,layout,docx_renderer,fit_check}.py`, `scripts/render_layout.py`, `tests/test_layout.py`
(16 test pass). Kiểm tra bằng mắt qua `compare.png` (scan | Word render) + sai số vị trí dọc từ PDF của Word.

| Mẫu | Kết quả | Lệch dọc TB (trước → sau vòng kín) |
|---|---|---|
| Công văn `aec455ba` | 1 trang; letterhead 2 cột, tiêu ngữ đậm + gạch chân, "ĐỀ XUẤT" giữa, bảng ô gộp đúng cột/cỡ/ngắt dòng, "Nơi nhận" đậm nghiêng, chữ ký + "Đồng ý" viết tay đúng chỗ | 2.0 → **0.0 pt** |
| Danh sách `9d6a89f4` | 1 trang; tiêu đề, dòng nghiêng, bảng 11×4 đúng cỡ, cột "Họ tên" căn trái | 3.7 → **0.0 pt** |
| Ảnh chụp `e57c5aec` | **Không dùng được** (tràn 2 trang, chữ giấy nền, cỡ chữ sai) → cần bước 1b | 50 pt (không hội tụ) |

DOCX: không có textbox (chữ sửa được, python-docx đọc 39 đoạn); ảnh nổi chỉ cho mực màu, con dấu, gạch trang trí.
Còn thiếu: kiểu chữ trong đoạn (nghiêng/đậm từng cụm); ngắt dòng có thể lệch 1 từ; gạch dưới "KHOA CN&ATTT" bị lọc mất.

### Flow hoàn chỉnh 1 lệnh + công thức (2026-11-24)

`python scripts/convert.py Testcase -o Testcase/output` (`src/conversion/pipeline.py`, `scripts/run_pipeline.py` đã bỏ).
Chạy lại từ đầu trên 4 ảnh: **1 phút 29 giây** (MinerU 20 s gồm load, sửa dấu 12 s gồm load, bố cục + Word 55 s).

| Mẫu | Sửa dấu | Trang Word | Lệch dọc TB | Ghi chú |
|---|---|---|---|---|
| Danh sách `9d6a89f4` | 0 | 1 | 0.0 pt | |
| Công văn `aec455ba` | 10 | 1 | 1.0 pt | |
| Ảnh chụp `e57c5aec` | 13 | 1 | 22.5 pt | Vẫn không dùng được — cần bước 1b (khung giấy + nắn phối cảnh) |
| **Sách toán** `Screenshot …` (mẫu mới) | 2 | 1 | 0.1 pt | Công thức Word thật (x², ℝ, phân số), lưới a)–f) đúng tab |

Lỗi chữ còn thấy trên sách toán là lỗi OCR: "R" thay "ℝ", "bồi" thay "bởi", "Tím Ảnh" thay "Tìm ảnh".
CER chưa đo được lần này: các file `Testcase/*.gt.txt` không còn trong thư mục.
Cài thêm vào môi trường chính: `latex2mathml` (LaTeX → MathML). Test: 22 pass.

### Pipeline theo lô, chỉ ghi .docx (2026-11-24)

`python scripts/convert.py <ảnh/thư mục...> -o <thư mục | file.docx> [--passes 2|1|0] [--keep-work <dir>]`.
Không còn thư mục `output/<stem>/...`: trung gian ở thư mục tạm, tự xoá. Kết quả mẫu: `Testcase/ket_qua/*.docx`.

| 4 ảnh Testcase | Tổng | MinerU (gồm load) | Sửa dấu | Bố cục | Word |
|---|---|---|---|---|---|
| Trước (mở Word 16 lần, warm-up MinerU, load tuần tự) | 89 s | 20 s | 12 s | — | 55 s |
| `--passes 2` (mặc định) | **33.6 s** | 16.7 s | 4.8 s | 2.8 s | 9.0 s |
| `--passes 1` | 29.1 s | | | | 4.6 s |
| `--passes 0` | 25.0 s | | | | 0 |

Lệch dọc với `--passes 2`: 0.1 / 1.0 / 22.5 (ảnh chụp) / 0.1 pt — bằng chất lượng cũ.
Phần cố định mỗi lần chạy ≈ 15 s (load MinerU); mỗi ảnh thêm ≈ 3–4 s → lô lớn nhanh hơn nhiều theo ảnh.
Muốn bỏ cả 15 s load: chạy MinerU thành server thường trực (`mineru-kit api-server`, CLAUDE.md mục 8) — chưa làm.
Số chỗ sửa dấu của ảnh chụp dao động giữa các lần chạy (9–13): kết quả OCR của MinerU trên ảnh xấu không ổn định.

### Sửa tổng quát sau 3 mẫu mới (2026-11-24) — hồi quy trên **cả 7 mẫu**
Người dùng thêm `Mau-cong-van-chi-dao-cap-nhat-nam-2022.jpg`, `cong-van-jpg-1475651904.webp`, `images.jpg` → lỗi bố cục.
Nguyên nhân gốc là các **ngưỡng tuyệt đối** chỉ đúng với mẫu cũ; đã thay bằng quy tắc **tương đối theo chính trang đó**:

| Lỗi | Nguyên nhân gốc | Sửa (tổng quát) |
|---|---|---|
| Quốc hiệu + tên cơ quan 2 cột bị gộp 1 dòng | Ngưỡng khe trắng cố định; chiều cao dòng bị dấu làm phồng | Tách tại khe **rộng nhất ở vùng giữa trang** (30–70%), ≥ 3× khe trung vị và ≥ 1 chiều cao chữ (`letter_h`, không tính dấu); 2 mảnh phải nằm 2 nửa trang. Gạch chân ngắn không tính là chữ |
| Ảnh chụp ám màu → chữ in bị coi là mực màu → thành ảnh | Chroma tuyệt đối | **Lệch màu so với màu mực của chính trang** (`ink_reference`): in ≤ 7.4, in dưới con dấu 13.9, bút/dấu 22–59 → ngưỡng 16 |
| Đoạn văn trên ảnh cong bị vỡ thành dòng căn giữa | Đoán "đoạn chảy" theo độ đầy dòng | Theo loại block MinerU (`text`/`list` nhiều dòng, ≥ 4 từ/dòng, không phải khối căn giữa ngắn) |
| Tên người ký xếp dọc → tràn trang 2 | Dòng độc lập bị Word ngắt dòng | Dòng độc lập rộng theo độ rộng chữ thật × 1.04, không bao giờ ngắt |
| Nét bút "/" bị OCR thành chữ 36 pt → đẩy cả trang xuống | Không có | Chữ ≤ 2 ký tự cỡ > 1.8× cỡ thường → ảnh nổi |
| Ảnh lệch, nền không đều | Không có tiền xử lý | `preprocess.py`: làm phẳng nền + deskew (±5°) + phóng to 3200 px |
| Dòng cuối tràn sang trang 2 | Bảng / line spacing exact cộng dồn | Vòng kín phạt tràn trang (`OVERFLOW_PENALTY`), co dòng 4–15% cho riêng tài liệu bị tràn, giữ lần tốt nhất |

Hồi quy toàn bộ (`python scripts/convert.py Testcase -o out --keep-work work`, `--passes 2`): **7/7 file, đều 1 trang, 66 s**.

| Mẫu | Trang | Lệch dọc TB (pt) | Trước khi sửa |
|---|---|---|---|
| 9d6a89f4 (bảng) | 1 | 0.1 | 0.1 |
| aec455ba (công văn) | 1 | 0.4 | 1.0 |
| e57c5aec (ảnh chụp) | 1 | 1.1 | 22.5 |
| Screenshot (công thức) | 1 | 0.1 | 0.1 |
| cong-van-jpg (mới) | 1 | 4.7 | đoạn văn thành ảnh |
| images.jpg (mới) | 1 | 7.0 | 2 trang / 23.4 |
| Mau-cong-van (mới) | 1 | 5.5 | 2 trang / 9.0 |

Còn hạn chế (không chỉnh ngưỡng riêng cho từng mẫu để tránh overfit):
- "Nơi nhận" nghiêng cỡ nhỏ mất đậm (độ đen nằm sát ngưỡng).
- Chữ in rất mờ ("BẠC LIÊU" trên `cong-van`) tách cột chưa đúng.
- Ảnh chụp mờ / có bìa nhựa (`images.jpg`): OCR sai chữ, một số dòng lệch — giới hạn ở đầu vào, không phải bố cục.
- MinerU gắn `paragraph_title` cho "Kính gửi" → bị in đậm; nghiêng một số dòng ngày tháng bị sót; chưa có kiểu chữ trong đoạn.

### Đầu vào PDF (2026-11-24) — ảnh hoặc PDF đều chạy chung 1 lệnh
`python scripts/convert.py <ảnh / PDF / thư mục...> -o <thư mục | file.docx>`: 1 DOCX / file, 1 trang Word / trang PDF.
Mỗi trang PDF phân loại riêng (`page_classifier.py`, CLAUDE.md mục 4) rồi đi đúng đường:

| Loại trang | Đường đi | Ghi chú |
|---|---|---|
| `scan` (ảnh cả trang, lớp OCR ẩn của máy scan, chữ lỗi mã) | render đúng độ phân giải ảnh gốc → làm phẳng + deskew + 3200 px → MinerU `standard` → sửa dấu | lớp OCR ẩn bị bỏ (mẫu Canon: "THANH Ho Ho cui tvtrNu") |
| `text_complex` (có công thức) | render vector 3200 px → MinerU `standard` → OMML | |
| `text_simple` | `pymupdf_extractor.py`: chữ / font / cỡ / đậm / nghiêng / màu / baseline đọc thẳng từ PDF, bảng `find_tables` (ô gộp), ảnh, đường kẻ | không GPU, ~ms; PDF chỉ có trang chữ → không khởi động MinerU |

Kết quả (14 file / 17 trang, 1 lệnh, `--passes 2`): **14/14 thành công, 114 s**. Mẫu PDF: `_data_soytehcm…pdf` (scan 2 trang thật),
`text_doc.pdf` (Word → PDF: quốc hiệu 2 cột, đoạn justify thụt đầu dòng, chữ đỏ nghiêng, bảng ô gộp), `mixed.pdf`
(1 trang chữ + 2 trang scan), `math.pdf` (công thức). 3 PDF thử nằm ngoài repo (sinh từ dữ liệu Testcase).

| File | Trang | Lệch dọc (pt) |
|---|---|---|
| text_doc.pdf | 1 text_simple | **0.1** |
| math.pdf | 1 text_complex | **0.1** (công thức thành equation Word) |
| _data_soytehcm (scan thật) | 2 scan | 16.5 → **8.9** |
| mixed.pdf | 1 text + 2 scan | 12.0 → **6.5** |

Sửa tổng quát phát hiện nhờ mẫu PDF + 3 ảnh chụp màn hình mới (chạy lại cả 10 ảnh, không mẫu nào xấu đi):
- `fit_paragraph` chỉ khớp số dòng + độ rộng **dòng cuối** → nhiều cỡ cùng thoả, cỡ nhảy 11.5 ↔ 14 pt khi khổ trang lệch 0.1 pt.
  Nay khớp **độ rộng mọi dòng** (images.jpg 6.9 → 5.7, Screenshot 150354 7.8 → 2.9).
- Khối căn giữa nhiều dòng (trích yếu "V/v …"): lề trái lệch nhau nhưng tâm dòng thẳng hàng → từng dòng đặt riêng, không gộp đoạn.
- XY-cut tách cột theo **khung mực đo được**, không theo khung đã nới cho vừa chữ (nới làm khe giữa 2 cột quốc hiệu < 4 pt → dính).
- Chia chữ cho các mảnh dòng: quy hoạch động giữ cỡ chữ suy ra của các mảnh gần nhau nhất (trước: làm tròn từng điểm cắt
  theo tỉ lệ độ rộng → "NAM" của "VIỆT NAM" rơi sang cột trái). Screenshot 150335: 22.2 → **3.2** pt.
- Crop con dấu: chỉ xoá nét đen nằm **trong dòng chữ đã dựng lại**; chữ ký mực đen đè lên dấu được giữ.

Còn hạn chế: chưa có kiểu chữ trong đoạn (cụm nghiêng / đậm); "Đính kèm" nghiêng cỡ nhỏ bị mất nghiêng; số viết tay
("687") làm dòng "Số:" to hơn bản gốc; ô bảng gộp trong PDF chữ căn giữa dọc thay vì căn trên; hình vẽ vector
(biểu đồ) trong PDF chữ chưa được giữ (chỉ đường kẻ mảnh); chữ xoay trong PDF chữ bị bỏ.

### Trang viết tay toàn bộ (2026-11-24) — 3 ảnh chụp vở ghi Transformer (229a…, 5acb…, 76ac…)
Trước: DOCX **toàn ảnh** (229: 0 ký tự, 40 ảnh, 5 MB). Nguyên nhân: màu mực chuẩn của trang tính trên mọi điểm tối của
ảnh, gồm cả **mặt bàn gỗ** quanh tờ giấy → màu chuẩn = nâu cam → mọi nét bút xanh lệch ≥ 16 → "mực màu / chữ ký" → ảnh.
Sửa tổng quát (CLAUDE.md 7.1 "Trang viết tay", "Công thức"):
- Màu mực chuẩn chỉ đo trong block chữ; véc-tơ màu chuẩn ≥ 10 → trang viết tay → giữ chữ. 12 trang cũ: màu chuẩn và số
  block "mực màu" **không đổi** (in ≤ 6.4).
- Công thức cao (ma trận, phân số): line spacing at-least, gộp hàng, ngoặc co giãn (`mfenced` → `m:d`), độ rộng ma trận
  hiệu chỉnh bằng Word (Score 3×3: Word 174 pt / ước lượng 176; WQ·WK·WV: 361 / 371).
- Lỗi có sẵn trong renderer: ô bảng ẩn bắt đầu bằng bảng con → đoạn văn đầu của ô (nằm **trên** bảng con) bị dùng cho
  đoạn đến sau → dòng "Chia từng phần tử…" nhảy lên đầu trang. Sửa + test.
- Dòng viết nghiêng bị đo thành nhiều mảnh → cả công thức dồn vào mảnh đầu (cỡ 6 pt) → gộp mảnh nối tiếp.

Kết quả (`convert.py Testcase`, 14 file / 15 trang, `--passes 2`, 188 s): 229 → 1 trang chữ + công thức, lệch dọc 21 pt
(khối 2 cột đầu trang cao hơn bản gốc, dồn xuống phần dưới); 5acb 1.0 pt; 76ac 3.5 pt nhưng hàng cuối bảng "V - Value"
tràn sang trang 2. 11 mẫu cũ giữ nguyên (0.0–8.9 pt; images.jpg 5.7 → 6.4 do OCR chạy lại khác chữ).
Chữ viết tay vẫn OCR sai nhiều (VLM đọc "Vì sao phải scale" → "Vi soo thai scale"): đây là giới hạn OCR, không phải bố cục.

### VLM đọc lại chữ viết tay (2026-11-27) — `docs/handwriting-ocr/DESIGN.md`, `src/conversion/handwriting.py`
Bước mới `[3/6] handwriting`: MinerU giữ bố cục, Qwen3.5-9B Q6_K (`tools/llama.cpp/llama-server.exe`,
`models/qwen3.5-9b/`, gitignored) đọc lại từng khối viết tay, 4 slot song song. Server khởi động cùng lúc MinerU.
Cờ `--hw-ocr auto|off|force`; thiếu exe / model → tự bỏ qua (MinerU như cũ). ProtonX bỏ qua span `hw`.
Đo trên RTX 5070 Ti, `--passes 2`:

| Lô | Vùng gửi / nhận / fallback / review | Bước handwriting | Cả lô | VRAM đỉnh (nền) |
|---|---|---|---|---|
| 3 trang vở | 38 / 36 / 2 / 11 | 10.9 s (≈ 0.29 s/vùng) | 57 s | **12.8 GiB** (1.6) |
| Cả Testcase, 14 file / 15 trang | 40 / 36 / 4 / 11 | 14.0 s | 137 s | **14.1 GiB** (1.6) |

- Chữ vở tốt hơn rõ ("Vì sao phải scale", "Ma trận sau scale", "Cộng tất cả lại", cả trang 5acb). Còn sai: viết tắt
  riêng, chữ xấu ("Mô hình khố học"), ô bảng viết tay (T5 chưa làm: 76ac "Tôi stag tìm ai").
- Trang in **không đổi**: lần chạy đầu 6/7 khối T1 là chữ trong **con dấu đỏ** → mực đỏ trên trang in giữ ảnh;
  ghi chú "Đồng ý" + nét ký (MinerU "Amy", VLM "Đặng") → T1 mà 2 bản đọc lệch nhiều thì giữ ảnh. |Δy| 11 mẫu in 0.0–8.9 pt như trước.
- Lệch thiết kế: kiểm tra độ dài so với chữ MinerU [0.33, 3] (ngưỡng bbox [0.3, 3] loại nhầm 13/26 bản đúng);
  thêm "LaTeX ngoài $", "chữ viết thành biến toán"; `X_{proj}` thiếu `$` → công thức; khối công thức cao không gửi VLM; bỏ T3.
- Test: `tests/test_handwriting.py` (46 test cả bộ qua).

### Cấu trúc `middle.json` 4.x (đã xác nhận)
- `pages[].blocks[]`: `type`, `index` (thứ tự đọc), `bbox` **chuẩn hoá 0..1 theo kích thước trang** `[x0, y0, x1, y1]`, `content`.
- `type` gặp được: `header`, `doc_title` (`level`), `paragraph_title` (`level`), `text`, `list` (`sub_type`, block con),
  `table` (`content[].type == "table_body"`, HTML có `rowspan`/`colspan`), `image` (`sub_type`: `text_image`, `seal`).
- **Không có** cỡ chữ, font, đậm/nghiêng, căn lề, điểm tin cậy → phải ước lượng từ bbox / ảnh.

## Next steps
Đã xong: phóng to + `standard` (chính tả), sửa dấu ProtonX, ground truth 3 mẫu, dựng DOCX giữ bố cục + vòng kín.
1. **Ảnh chụp**: đã có làm phẳng nền + deskew (`preprocess.py`); còn thiếu tìm khung giấy + nắn phối cảnh.
2. **Kiểu chữ trong đoạn**: nghiêng / đậm theo từng cụm từ (cần vị trí từng từ: căn chữ OCR vào các nét trên ảnh).
3. Gạch dưới "KHOA CN&ATTT" bị lọc mất; ngắt dòng trong đoạn lệch 1 từ (thử chỉnh lề phải theo vòng kín).
4. PDF: đã có (phân loại từng trang + extractor PyMuPDF). Còn: hình vẽ vector, chữ xoay, `flash` cho trang công thức,
   căn dọc ô bảng, PDF thật có chữ (mới thử PDF sinh từ Word).
5. Ghép thành 1 lệnh `python -m conversion <file> -o <out.docx>`; `cache.py`.
6. Validate + OCR lại vùng hỏng (CLAUDE.md 5.6) nếu còn lỗi chữ; benchmark; xoá code YOLO / Qwen2.5-VL, cập nhật
   `requirements.txt`, `README.md`.

## Decisions / gotchas
- Bố cục: **viết renderer mới** (`conversion/layout.py` + `docx_renderer.py`) đọc IR, **mang sang thuật toán** của
  Step 8 cũ (`processs_pdf_to_docs.py`: nhóm hàng theo Y, bảng ẩn cho khối song song, indent theo x, spacing theo Y-gap,
  `determine_alignment_by_position`). Không sửa tiếp hàm cũ 1300 dòng.
- MinerU renderers bỏ block `header` → luôn render từ `middle.json`, không từ markdown/DOCX của MinerU.
- Trang scan tiếng Việt: mặc định `standard`, không phải `basic` (PP-OCR của `basic`/`flash` mất dấu).
- Tier cao hơn không luôn tốt hơn (`advanced` làm rơi dòng) → chấm điểm từng block, giữ block tốt nhất.
- Không đặt chữ trong textbox (DOCX phải sửa được; bảng/công thức là đối tượng Word thật).
- MinerU chạy trong `.venv-mineru/` riêng; dự kiến server port 8001 (VLM 30000).
- Không đụng `src/llm_pipeline/`, `api/`, chatbot cho tới khi người dùng yêu cầu.
- Windows: đặt `PYTHONUTF8=1` khi chạy MinerU CLI/script.
- `Testcase/` có dữ liệu thật → gitignored, không commit.
- Bố cục: **mọi ngưỡng phải tương đối theo chính trang** (khe trung vị, màu mực trang, chiều cao chữ không dấu);
  mỗi lần sửa chạy lại **cả** `Testcase`, không chỉnh riêng cho 1 mẫu.
