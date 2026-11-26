# Thiết kế: MinerU + VLM đọc chữ viết tay

Ngày: 2026-11-26 (sửa: trạng thái thật của các tín hiệu phát hiện; **đã hiện thực 2026-11-27**, chỗ lệch thiết kế ghi ở từng mục) · Áp dụng cho nhánh `minerU` của `pdf_processing` · Số liệu lấy từ [BENCHMARK.md](BENCHMARK.md) trong folder này.

## 1. Tóm tắt quyết định

- **MinerU vẫn làm toàn bộ bố cục**: khối chữ, bảng, công thức, ảnh, thứ tự đọc. VLM không thay MinerU.
- **VLM (Qwen3.5-9B GGUF Q6_K qua `llama-server`) chỉ đọc lại những vùng có chữ viết tay.**
  Đọc từng vùng cắt nhỏ, không đọc cả trang.
- **Trang in thì không gọi VLM.** Tốc độ trang in giữ nguyên như hiện nay.
- **Nếu VLM lỗi hoặc output không qua kiểm tra, giữ nguyên hành vi hiện tại** (chữ của MinerU, hoặc ảnh cắt).
  Vì vậy bật luồng này không làm kết quả xấu đi trên những trang đang làm tốt.

Lý do (số đo trên 37 vùng viết tay, 3 trang vở):

| | CER | WER | s/vùng (T4) | VRAM |
|---|---|---|---|---|
| Qwen3.5-9B GGUF Q6_K | 7.9% | 15.0% | 1.5 tuần tự · ~0.55 khi chạy 4 slot | 8.8 GiB |
| Qwen3.5-9B GGUF Q4_K_M | 8.9% | 16.5% | 1.1 · ~0.43 | 7.4 GiB |
| Qwen2.5-VL-7B fp16 (hướng ban đầu) | 9.7% | 18.1% | 4.7 | 15.5 GiB |
| Vintern-1B-v3.5 | 15.7% | 27.4% | 5.1 | 1.8 GiB |

Chưa đo chính MinerU trên bộ 37 vùng này (xem mục 9).

## 2. Luồng tổng thể

```
[1/6] input      PDF / ảnh → page_classifier
                  ├─ PDF có lớp text (digital)  → PyMuPDF, KHÔNG OCR, KHÔNG VLM
                  └─ ảnh / scan                 → preprocess (flatten, deskew, 3200 px)
[2/6] ocr        MinerU standard → middle.json            (đồng thời: load ProtonX + khởi động llama-server)
[3/6] handwriting  MỚI: chọn vùng viết tay → cắt → VLM đọc song song → kiểm tra → ghi vào middle.json
[4/6] spelling   ProtonX sửa dấu, BỎ QUA các span do VLM viết
[5/6] layout     build_page (đo, dàn trang)
[6/6] word       closed-loop qua Word
```

Bước 3 là bước duy nhất được thêm vào. Bước này chạy trên `u.fixed` (bản sao của `middle.json`) trước bước spelling,
nên layout và renderer không cần biết chữ đến từ MinerU hay từ VLM.
Ngoại lệ là cờ `hw` dùng ở mục 5.4.

## 3. Khi nào dùng VLM, khi nào chỉ dùng MinerU

### 3.1 Theo trang

| Loại trang | Nhận biết | Xử lý |
|---|---|---|
| PDF digital | `page_classifier` (đã có) | PyMuPDF. Không MinerU, không VLM |
| Scan in máy, không có chữ viết tay | không có vùng nào qua mục 3.2 | **Chỉ MinerU** (+ ProtonX) |
| Form / văn bản in có điền tay, có chữ ký | có vùng qua mục 3.2 | MinerU + VLM **chỉ ở những vùng đó** |
| Trang viết tay mực màu (vở, ghi chú) | `is_pen_page(ink_ref)` đúng | MinerU để lấy bố cục. VLM đọc lại **mọi khối chữ** |
| Trang viết tay mực đen | **chưa có cách nhận biết** (T4, mục 3.2) | Tạm thời: như trang in, trừ khi người dùng chạy `--hw-ocr force` |

### 3.2 Theo vùng: gửi một khối MinerU sang VLM khi có ít nhất một tín hiệu

Trạng thái thực tế trong code (kiểm tra 2026-11-26):

| # | Tín hiệu | Trạng thái | Cách nhận biết |
|---|---|---|---|
| T1 | Khối chữ mực màu | **Đã có** | `measure_block` gắn `colored_ink = True` khi trung vị `hue_offset(pixel, ink_ref) ≥ 16` (`measure._ink_color`). Số đo: in ≤ 7.4, in dưới con dấu 13.9, bút/dấu 22–59. Hiện tại khối này bị cắt thành ảnh, đây là nơi VLM được lợi nhiều nhất |
| T1 (bổ sung 2026-11-27) | Trừ mực **đỏ** trên trang in | **Đã có** | Chạy thử: 6/7 khối T1 trên trang in là **chữ trong con dấu đỏ** ("HỎA TỐC", vòng chữ dấu tròn) mà MinerU không đóng khung `seal`. Đỏ (R lớn nhất trong trung vị màu mực) → giữ ảnh như cũ. Bút bi xanh vẫn gửi VLM. Trang vở (T2) không áp dụng: tiêu đề bút đỏ vẫn là chữ |
| T2 | Trang viết tay | **Đã có, chỉ cho mực màu** | `ink_reference` = trung vị véc-tơ màu (rgb − xám) của mực **trong các khối chữ MinerU**. `is_pen_page`: `‖ink_ref‖ ≥ 10` (in ≤ 6.4 trên 12 trang, bút xanh 13.1–14.5). Trang vở **mực đen** có `‖ink_ref‖` ≈ chữ in → không bắt được |
| T3 | MinerU báo ảnh có chữ (`text_image`) | **Bỏ** (2026-11-27) | MinerU 4.2.3 chỉ trả `sub_type: seal` trên cả 14 trang mẫu, không có `text_image` |
| T4 | Viết tay mực đen | **Chưa có. Cần đo** | Xem 3.3. Kiểm tra "chữ rác" (`page_classifier._is_garbled`: chỉ `�`, PUA, ký tự điều khiển) **không** dùng được: MinerU đọc sai chữ viết tay ra tiếng Việt trông hợp lý ("Giữ" → "Gửi"), không ra rác |
| T5 | Ô bảng có chữ viết tay | **Chưa có** | Xem 3.4 |

### 3.3 T4: phát hiện viết tay mực đen (đề xuất, xếp từ rẻ đến tốn)

Không dùng regex hay danh sách âm tiết. `CLAUDE.md` mục 5.5 (1b) và 5.6 đã chốt bỏ cách này.

| Cách | Đo gì | Chi phí | Ghi chú |
|---|---|---|---|
| **G. Hình học dòng** | Từ các dòng `measure_block` đã đo: độ lệch chuẩn tương đối của chiều cao dòng, độ dốc / độ cong baseline, độ đều độ đen lõi nét giữa các dòng, góc nghiêng chữ (đã có phép xiên 12° cho chữ nghiêng). Chữ in đều, chữ viết tay dao động | ~0 (số đã đo sẵn) | Ngưỡng phải **tương đối theo trang** (quy tắc chung `CLAUDE.md` 7.1) |
| **S. Tín hiệu 5.6 có sẵn trong kế hoạch** | S1: surprisal từ LM tiếng Việt. S2: hai lần đọc không khớp | 1 forward pass / trang + đọc lại vùng bị gắn cờ | Không nhận ra viết tay, mà nhận ra **chữ đọc sai**. Vẫn hữu ích: vùng bị S1 + S2 gắn cờ thì gửi VLM |
| **C. Classifier in / viết tay** | CNN nhỏ (ONNX, vài MB) trên ảnh từng dòng | vài ms / dòng | Cần dữ liệu train: vùng cắt in và viết tay từ `testcase/` + bộ công khai (UIT-HWDB) |
| **V. Hỏi VLM có/không** | "Vùng này là chữ in hay viết tay?" với `max_tokens 1` | ~0.3–0.5 s / khối | Chỉ dùng cho khối G/C còn phân vân |

Đề xuất: **G trước, V cho khối phân vân**, nếu G không đủ thì mới làm C.
Tiêu chí nhận: trên các trang in hiện có, gắn nhầm < 2% khối chữ in. Trên trang viết tay mực đen, bắt ≥ 90% khối viết tay.
Cần có trang vở hoặc form viết bằng bút đen để đo, hiện chưa có.

### 3.4 T5: ô bảng có chữ viết tay (đề xuất)

MinerU chỉ trả HTML của bảng, **không có bbox từng ô**.

1. **Tìm khung ô:**
   - Bảng có kẻ: `layout._rules(gray, "h"/"v")` đã tìm vị trí đường kẻ. Giao các đường này cho ra lưới ô.
     Ghép với `_grid_shape(html)` (số hàng/cột, có `rowspan`/`colspan`). Lưới đo được lệch số hàng/cột của HTML thì bỏ qua bảng đó.
   - Bảng không kẻ: chưa có cách, để nguyên MinerU.
2. **Kiểm tra từng ô:** mực màu dùng `hue_offset ≥ 16` như T1. Mực đen dùng G/V của 3.3.
3. **Đọc lại** ô bị gắn cờ bằng VLM, rồi thay chữ vào đúng `<td>` trong HTML. Cấu trúc bảng giữ nguyên.

**Không** gửi sang VLM:

- Chữ in, gồm cả bảng in và tiêu đề. MinerU đọc chữ in tốt; VLM chỉ thêm thời gian và có nguy cơ "sửa" chữ đúng.
- Công thức: MinerU đã ra LaTeX/OMML.
- **Chữ ký, con dấu, hình vẽ, logo**: giữ ảnh như hiện tại. Nếu cần chữ, ghi `<SIGNATURE>`.
- Vùng quá nhỏ (ví dụ < 24 px chiều cao ở ảnh 3200 px) hoặc chỉ có 1–2 ký tự (dấu tích, gạch).
- Toàn bộ trang khi chế độ là `off` (mục 7).

Bộ phân biệt chữ ký với chữ viết tay (T1 nhưng không phải chữ) có thể làm rẻ như sau:
- Nét liền dài và không có khoảng trắng giữa các từ.
- Nằm gần nhãn in "Ký tên", "Chữ ký", "(ký, ghi rõ họ tên)".

Nếu chưa chắc, cứ gửi VLM. Prompt cho phép VLM trả `<SIGNATURE>`.

## 4. Gọi VLM

| Mục | Giá trị | Lý do |
|---|---|---|
| Model | `unsloth/Qwen3.5-9B-GGUF` **Q6_K** + `mmproj-F16` | CER ngang Q8_0, rẻ hơn 1.7 GiB. Dùng Q4_K_M nếu thiếu VRAM |
| Server | `llama-server --parallel 4 -fa on --jinja --reasoning-budget 0 -ngl 999` | 4 slot song song nhanh hơn khoảng 2.7 lần so với chạy tuần tự |
| Context | `-c 16384` (4 slot × 4096) | Ảnh vùng ≤ 1280 token. Không cần 8192/slot như lúc benchmark |
| Ảnh | Cắt theo bbox MinerU, nới 4–8 px. Thu nhỏ để ≤ 1280·28·28 pixel | Giống benchmark, giữ độ chính xác |
| Form | Cắt **cả nhãn in đứng trước** ô điền (ví dụ "Họ tên: ____") | Có ngữ cảnh nên đọc đúng hơn. Bỏ phần nhãn khỏi output bằng cách so với chữ MinerU đã đọc |
| Prompt | Prompt của benchmark, thêm: "Nếu là chữ ký hoặc không có chữ, trả `<SIGNATURE>` / `<EMPTY>`" | |
| Giải mã | `temperature 0`, `repeat_penalty 1.05`, `max_tokens ≈ 1.5 × số ký tự dự kiến` (từ bề rộng bbox / cỡ chữ) | Chặn việc model lặp vô tận |
| Thinking | Tắt | Bật thinking làm chậm gấp nhiều lần, không có lợi cho OCR |

Server được khởi động ở bước 2, song song với MinerU, giống cách ProtonX đang load.
Nếu server không lên trong N giây thì bỏ qua bước 3 và ghi log cảnh báo.

## 5. Ghép kết quả và kiểm tra

### 5.1 Kiểm tra output VLM (trượt bất kỳ mục nào thì dùng fallback)

1. Không rỗng. Nếu là `<SIGNATURE>` thì giữ ảnh.
2. Độ dài hợp lý. **Đã đổi khi hiện thực**: so với số ký tự MinerU đọc ở cùng khối, tỉ lệ trong [0.33, 3]
   (tương đối theo chính trang). MinerU không có chữ thì so với bề rộng bbox / chiều cao dòng × số dòng, trong [0.33, 6].
   Ngưỡng [0.3, 3] ban đầu loại nhầm 13/26 bản đọc đúng (chữ viết tay thật nằm ở 3.0–4.4 ký tự / chiều cao dòng).
   Thêm 2 kiểm tra: không có lệnh LaTeX (`rac`…) ngoài `$`; không viết chữ thường thành biến toán rời (`$s c a l e$`).
   `X_{proj}` viết thiếu `$` được tự chuyển thành công thức inline.
3. Không lặp: không có n-gram nào (n = 3) lặp quá 3 lần. Không chạm `max_tokens`.
4. Đúng hệ chữ: tỉ lệ ký tự thuộc khối Unicode IPA (U+0250–02AF) hoặc ngoài Latin / số / ký hiệu toán phải < 5%.
   Benchmark gặp lỗi model trả ký hiệu phiên âm `[mɛ̃] [ɑ̃]`, và mục này loại được lỗi đó.
   Đây là kiểm tra theo khối Unicode, không phải danh sách âm tiết. Kiểm tra nghĩa của chữ để cho S1 (`CLAUDE.md` 5.6).
5. Không phải lời giải thích: không bắt đầu bằng "Đây là", "Văn bản trong ảnh", "Here is".

### 5.2 Chọn giữa MinerU và VLM

| Tình huống | Chọn |
|---|---|
| MinerU và VLM gần giống nhau (CER giữa hai bản < 10%) | VLM (thường đúng dấu hơn) |
| Vùng viết tay (T1/T2/T3/T5), VLM qua kiểm tra | **VLM** |
| T4 (viết tay mực đen, theo 3.3), VLM qua kiểm tra | VLM |
| VLM trượt kiểm tra | MinerU nếu chữ MinerU hợp lệ, nếu không thì **ảnh cắt** (hành vi hiện tại) |
| Khác nhau nhiều và cả hai đều hợp lệ | VLM, đánh dấu `needs_review` để báo cáo cuối liệt kê |
| Khác nhau nhiều, khối T1 trên trang in (bổ sung 2026-11-27) | **Giữ ảnh**: ghi chú ngắn "Đồng ý" + nét ký, MinerU đọc "Amy", VLM đọc "Đặng", cả hai sai |

### 5.3 Viết vào `middle.json`

Thay `content` của span bằng chữ VLM và thêm `"hw": true` (cùng `"src": "vlm"`).
Giữ nguyên bbox để layout đo như cũ.

Khối có công thức cao (phân số, ma trận: `height_em` > 1.5) hoặc chữ thường < 60% ký tự **không** gửi VLM:
LaTeX của MinerU cho các khối đó tốt hơn (đã thử: VLM viết lại phân số Scale thành chữ). Khối `equation` mà VLM đọc là chữ
được đổi thành `text`.

### 5.4 Spelling và layout

- **ProtonX bỏ qua span `hw`.** Model này được huấn luyện cho chữ in và văn bản pháp lý,
  nên dễ "sửa" mất từ viết tắt của người viết như `g/trị`, `k/thức`, `h/quả`.
  Ghi chú: hiện tại `correct_many` đã bỏ qua slot có `<`. Cần thêm điều kiện bỏ qua `hw`.
- Layout: một khối `colored_ink` mà VLM đọc được thì đổi thành **chữ** (có thể giữ màu mực làm màu run) thay vì `FloatImage`.
  Nếu không đọc được thì vẫn là ảnh.

## 6. Tốc độ và VRAM (16 GB)

| | VRAM | Ghi chú |
|---|---|---|
| Qwen3.5-9B Q6_K + mmproj f16, ctx 4×4096 | +8.4–8.7 GiB (đo trên 5070 Ti) | server sẵn sàng sau 6–7 s, chạy song song MinerU |
| MinerU + ProtonX + llama-server cùng lúc | **đỉnh 12.8 GiB** (nền 1.6) | Đo 2026-11-27, 3 trang vở; vừa 16 GB, không cần chạy tuần tự |
| ProtonX (protonx-legal-tc) | ~0.9 GiB | Số đo 2026-11-24 (`CLAUDE.md` 5.5) |

Thời gian thêm vào mỗi trang ≈ số vùng × ~0.55 s (T4, 4 slot). **Đo trên 5070 Ti (2026-11-27)**: 38 vùng / 3 trang vở
trong 10.9 s (≈ 0.29 s/vùng, 4 slot), cả lô 57 s gồm Word 2 vòng.

- Trang in: +0 s.
- Form có khoảng 5 ô điền tay: khoảng +3 s.
- Trang vở khoảng 13 khối: khoảng +7 s.

Nếu MinerU, ProtonX và Q6_K cộng lại vượt 16 GB:
1. Dùng Q4_K_M (giảm 1.4 GiB, CER tăng khoảng 1 điểm).
2. Hoặc chạy tuần tự: MinerU xong thì giải phóng VRAM rồi mới bật llama-server.
   Batch nhiều trang thì chi phí bật server chỉ trả một lần.

Muốn nhanh hơn nữa:
- Gom các vùng của **tất cả trang** trong batch vào một hàng đợi, để 4 slot luôn có việc.
- Cache theo hash ảnh vùng, để chạy lại không phải đọc lại.
- Vùng là một dòng ngắn thì đặt `max_tokens` thấp.

## 7. Cấu hình

```
--hw-ocr {auto,off,force}   mặc định auto
    auto   chỉ các vùng qua mục 3.2
    off    hành vi hiện tại (chỉ MinerU)
    force  đọc lại mọi khối chữ bằng VLM (để debug / so sánh)
--hw-model PATH             GGUF, mặc định models/Qwen3.5-9B-Q6_K.gguf
--hw-mmproj PATH            mặc định models/mmproj-F16.gguf
--hw-slots N                mặc định 4
```

Mỗi trang ghi vào báo cáo: số vùng gửi VLM, số vùng được chấp nhận, số vùng fallback, số vùng `needs_review`, và thời gian.

## 8. Vị trí trong code (đã làm 2026-11-27; test: `tests/test_handwriting.py`)

| File | Thay đổi |
|---|---|
| `src/conversion/handwriting.py` (mới) | `find_regions(middle_page, image) -> list[Region]` (mục 3.2). `read_regions(regions, client)` (mục 4). `accept(region, text) -> bool` (mục 5.1) |
| `src/conversion/pipeline.py` | Khởi động `llama-server` ở bước 2. Thêm bước `[3/6] handwriting`. Đánh số lại log |
| `src/conversion/spelling.py` | `_slots`: bỏ qua span có `hw` |
| `src/conversion/layout.py` | `build_page`: khối `colored_ink` có span `hw` được dàn như chữ, không thành `FloatImage` |
| `scripts/convert.py` | Các cờ `--hw-*` |

`ink_reference` và `hue_offset` là hàm thuần trong `measure.py`, nên bước 3 dùng lại được trước bước layout.

## 9. Rủi ro và việc cần đo tiếp

1. **Chưa có baseline MinerU trên 37 vùng.** Cần chạy MinerU trên 3 trang vở rồi chấm CER/WER cùng cách.
   Đây là con số quyết định VLM thắng MinerU bao nhiêu trên chữ viết tay.
2. Bộ đo nhỏ: 1 người viết, 407 từ, chỉ có vở ghi.
   Cần thêm form điền tay, chữ của nhiều người, mực đen (để thử T4), ảnh chụp điện thoại.
3. Đo tốc độ và VRAM thật trên RTX 5070 Ti / Windows: MinerU, ProtonX và llama-server chạy cùng lúc.
4. T4 (viết tay mực đen) chưa có cách nhận biết và chưa có dữ liệu đo. Tiêu chí nhận ở mục 3.3.
5. Lỗi chung của mọi model: chữ viết tắt riêng của người viết ("o" có gạch = "không").
   Muốn hết lỗi này cần một bảng viết tắt hoặc fine-tune QLoRA trên chữ của chính người dùng.
6. Lỗi đọc thành ký hiệu phiên âm `[mɛ̃]` chỉ xuất hiện khi chạy llama.cpp, không có khi chạy transformers.
   Cần kiểm tra cách llama.cpp tiền xử lý ảnh, hoặc nới vùng cắt.
7. T3 (`text_image`) chưa kiểm chứng là MinerU 4.2.3 có trả hay không (mục 3.2).
8. T5 cần lưới ô từ đường kẻ. Bảng không kẻ chưa có cách tìm khung ô.
