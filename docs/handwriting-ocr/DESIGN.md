# Thiết kế: MinerU + VLM đọc chữ viết tay

Ngày: 2026-11-26 · Áp dụng cho nhánh `minerU` của `pdf_processing` · Số liệu lấy từ [BENCHMARK.md](BENCHMARK.md) trong folder này.

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
| Trang viết tay (vở, ghi chú) | `is_pen_page(ink_ref)` đúng (mực màu chiếm cả trang) | MinerU để lấy bố cục. VLM đọc lại **mọi khối chữ** |

### 3.2 Theo vùng: gửi một khối MinerU sang VLM khi có ít nhất một tín hiệu

| # | Tín hiệu | Lấy từ đâu | Ghi chú |
|---|---|---|---|
| T1 | Khối chữ có `colored_ink = True` (mực khác màu chữ in, `hue_offset ≥ 16`) | `measure_block` / `_ink_color` (đã có) | Hiện tại khối này bị cắt thành ảnh. Đây là nơi được lợi nhiều nhất |
| T2 | Trang là `pen_page` | `is_pen_page` (đã có) | Đọc lại mọi khối `text` / `title` / `list` |
| T3 | Khối `image` có `sub_type == "text_image"` | MinerU `middle.json` | MinerU thấy chữ trong ảnh nhưng không đọc |
| T4 | Chữ MinerU trả về trông như rác: tỉ lệ âm tiết không hợp lệ trong tiếng Việt cao, ký tự lạ, lặp | kiểm tra âm tiết + `page_classifier._is_garbled` | Bắt trường hợp viết tay **mực đen**: T1 không bắt được vì mực cùng màu chữ in |
| T5 | Ô bảng có mực viết tay (T1 hoặc T4 trên vùng của ô) | lưới `_grid_shape` + `hue_offset` theo ô | Chỉ đọc lại **chữ trong ô**. Giữ cấu trúc HTML của MinerU |

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
2. Độ dài hợp lý so với vùng: số ký tự / (bề rộng bbox / chiều cao dòng) nằm trong [0.3, 3].
3. Không lặp: không có n-gram nào (n = 3) lặp quá 3 lần. Không chạm `max_tokens`.
4. Đúng chữ: ≥ 80% âm tiết là âm tiết tiếng Việt hợp lệ, hoặc là số, ký hiệu toán, từ tiếng Anh.
   Benchmark bắt được lỗi model trả ký hiệu phiên âm `[mɛ̃] [ɑ̃]`. Mục này loại được lỗi đó.
5. Không phải lời giải thích: không bắt đầu bằng "Đây là", "Văn bản trong ảnh", "Here is".

### 5.2 Chọn giữa MinerU và VLM

| Tình huống | Chọn |
|---|---|
| MinerU và VLM gần giống nhau (CER giữa hai bản < 10%) | VLM (thường đúng dấu hơn) |
| Vùng viết tay (T1/T2/T3/T5), VLM qua kiểm tra | **VLM** |
| T4 (MinerU ra rác), VLM qua kiểm tra | VLM |
| VLM trượt kiểm tra | MinerU nếu chữ MinerU hợp lệ, nếu không thì **ảnh cắt** (hành vi hiện tại) |
| Khác nhau nhiều và cả hai đều hợp lệ | VLM, đánh dấu `needs_review` để báo cáo cuối liệt kê |

### 5.3 Viết vào `middle.json`

Thay `content` của span bằng chữ VLM và thêm `"hw": true` (cùng `"src": "vlm"`).
Giữ nguyên bbox để layout đo như cũ.

### 5.4 Spelling và layout

- **ProtonX bỏ qua span `hw`.** Model này được huấn luyện cho chữ in và văn bản pháp lý,
  nên dễ "sửa" mất từ viết tắt của người viết như `g/trị`, `k/thức`, `h/quả`.
  Ghi chú: hiện tại `correct_many` đã bỏ qua slot có `<`. Cần thêm điều kiện bỏ qua `hw`.
- Layout: một khối `colored_ink` mà VLM đọc được thì đổi thành **chữ** (có thể giữ màu mực làm màu run) thay vì `FloatImage`.
  Nếu không đọc được thì vẫn là ảnh.

## 6. Tốc độ và VRAM (16 GB)

| | VRAM | Ghi chú |
|---|---|---|
| Qwen3.5-9B Q6_K + mmproj f16, ctx 4×8192 | 8.8 GiB (đo trên T4) | ctx 4×4096 sẽ thấp hơn |
| MinerU standard | **chưa đo** | Cần đo trên 5070 Ti |
| ProtonX | **chưa đo** | |

Thời gian thêm vào mỗi trang ≈ số vùng × ~0.55 s (T4, 4 slot). Thời gian thật trên 5070 Ti chưa đo.

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

## 8. Vị trí trong code (đề xuất)

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
4. T4 (phát hiện viết tay mực đen) chưa có bộ đo riêng. Cần đo tỉ lệ bắt nhầm chữ in.
5. Lỗi chung của mọi model: chữ viết tắt riêng của người viết ("o" có gạch = "không").
   Muốn hết lỗi này cần một bảng viết tắt hoặc fine-tune QLoRA trên chữ của chính người dùng.
6. Lỗi đọc thành ký hiệu phiên âm `[mɛ̃]` chỉ xuất hiện khi chạy llama.cpp, không có khi chạy transformers.
   Cần kiểm tra cách llama.cpp tiền xử lý ảnh, hoặc nới vùng cắt.
