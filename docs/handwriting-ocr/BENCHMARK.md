# Handwriting OCR Benchmark (2026-11-26)

Dữ liệu: 3 trang vở ghi viết tay tiếng Việt (`pdf_processing/testcase/`), 37 vùng cắt, 407 từ ground truth gõ tay.
Máy: Kaggle GPU T4, greedy, cùng một prompt và cùng giới hạn pixel cho mọi model.
Bản thường chạy bằng transformers fp16 (2×T4); bản `GGUF` chạy bằng llama.cpp `llama-server` trên 1×T4, mmproj f16.

Bản này chỉ gồm kết quả và ghi chú. Folder benchmark đầy đủ (`compare.html`, output, ảnh vùng cắt, script, log) được giữ ở máy local, không commit vì ảnh lấy từ `testcase/`.

## Kết quả

| Model | CER vùng* | CER vùng (cả bảng) | WER* | Từ sai/thiếu | Đọc cả trang: tìm được từ | s/vùng (T4) | VRAM |
|---|---|---|---|---|---|---|---|
| **Qwen3.5-9B** fp16 | **7.1%** | **9.0%** | 14.0% | 13.0% | 84.7% | 15.0 | 17.6 GiB |
| Qwen3.5-9B GGUF Q8_0 | 7.9% | 9.8% | 15.0% | 14.2% | 85.4% | 1.44 | 10.5 GiB |
| **Qwen3.5-9B GGUF Q6_K** | 7.9% | 9.7% | 15.0% | 14.2% | 84.7% | 1.52 | 8.8 GiB |
| Qwen3.5-9B GGUF Q4_K_M | 8.9% | 10.8% | 16.5% | 16.0% | 85.2% | **1.09** | 7.4 GiB |
| Qwen3.5-9B GGUF UD-Q4_K_XL | 10.1% | 11.9% | 20.2% | 15.0% | 84.8% | 1.37 | 7.6 GiB |
| Qwen2.5-VL-7B | 9.7% | 9.5% | 18.1% | 17.2% | 78.3% | 4.7 | 15.5 GiB |
| Qwen3-VL-8B | 10.1% | 12.1% | 19.1% | 17.7% | 83.9% | 14.9 | 16.3 GiB |
| Qwen3-VL-4B | 14.7% | 16.7% | 31.3% | 26.5% | 78.7% | 12.9 | 8.3 GiB |
| Vintern-1B-v3.5 | 15.7% | 17.2% | 27.4% | 24.1% | 52.2% | 5.1 | 1.8 GiB |

WER = (từ bị thay + bị thiếu + bị thừa) / số từ ground truth, có tính thứ tự, nên khắt khe hơn cột "Từ sai/thiếu" (cột đó không phạt từ thừa và không xét thứ tự).
\* Không tính `p3_bang` (bảng 2 cột: đúng chữ, khác thứ tự đọc).
Thời gian của bản transformers bị chậm bất thường: trên Kaggle, Qwen3.5 và Qwen3-VL chạy bằng code PyTorch dự phòng vì thiếu `flash-linear-attention` / `causal_conv1d`.
Thời gian của bản GGUF là thời gian thật của llama.cpp trên T4. RTX 5070 Ti có băng thông bộ nhớ gấp khoảng 3 lần T4, nên dự kiến sẽ nhanh hơn nữa (chưa đo).

## Nội dung folder

| Đường dẫn | Nội dung |
|---|---|
| `compare.html` | Trang so sánh (tự chứa, ảnh nhúng sẵn) |
| `summary.csv` | Bảng tổng hợp ở trên |
| `comparison.csv` | Mỗi vùng 1 dòng: ground truth + output + CER của từng model |
| `outputs/<model>.json` | Output gốc của từng model (vùng + cả trang, thời gian) |
| `full_page_outputs/<model>/<trang>.txt` | Output khi đọc cả trang một lần |
| `crops/`, `ground_truth.json` | 37 ảnh vùng cắt và ground truth |
| `scripts/` | `bench_hw.py` (transformers, Kaggle), `bench_gguf.py` (llama.cpp GGUF, Kaggle), `make_crops.py`, `regions.json` (toạ độ + GT), `build_report.py` |
| `logs/` | Log Kaggle |

Dựng lại báo cáo: `python3 scripts/build_report.py`.

## Qwen3.5-9B: các bản quantize

Qwen không phát hành bản quantize chính thức cho Qwen3.5-9B. Mọi bản dưới đây đều do cộng đồng hoặc bên thứ ba làm.

| Định dạng | Ví dụ | Kích thước | Chạy bằng | Trên Windows + 5070 Ti |
|---|---|---|---|---|
| **GGUF Q4_K_M** | `unsloth/Qwen3.5-9B-GGUF`, `bartowski/Qwen_Qwen3.5-9B-GGUF` | ~5.8–6.2 GB + mmproj f16 0.9 GB | llama.cpp | **Khuyên dùng**: chạy native, cùng stack llama.cpp với MinerU |
| GGUF Q5_K_M / Q6_K | cùng repo | ~6.5–7.5 GB | llama.cpp | Nếu Q4 làm giảm độ chính xác rõ |
| GGUF Q8_0 | cùng repo | ~9.8 GB | llama.cpp | Gần như không mất độ chính xác, vẫn vừa 16 GB |
| GGUF MTP | `prithivMLmods/Qwen3.5-9B-MTP-GGUF` | ~Q4 | llama.cpp có MTP | Multi-token prediction: decode nhanh hơn; cần kiểm tra bản llama.cpp hỗ trợ |
| FP8 dynamic | `RedHatAI/Qwen3.5-9B-FP8-dynamic` | ~10 GB | vLLM | Cần WSL2 (vLLM không chạy native trên Windows) |
| W4A16 (GPTQ) | `RedHatAI/Qwen3.5-9B-quantized.w4a16` | ~6 GB | vLLM | Như trên |
| AWQ | `QuantTrio/Qwen3.5-9B-AWQ` | ~6 GB | vLLM | Như trên |
| NVFP4 | `RishabhSinha/Qwen3.5-9B-NVFP4`, … | ~6 GB | vLLM / TensorRT-LLM | Blackwell (5070 Ti) có phần cứng FP4, nhưng phần mềm trên Windows chưa sẵn |

Ghi chú khi chạy nhanh:
- Mỗi vùng chỉ cho ra 10–60 token, nên thời gian chủ yếu nằm ở **encode ảnh và prefill**, không phải decode.
  Vì vậy giới hạn số pixel của ảnh vùng và chạy song song (`llama-server --parallel N`) giúp nhiều hơn là hạ bit.
- Giữ **mmproj ở f16** (phần vision): quantize phần này dễ làm giảm độ chính xác OCR nhất.
- Tắt thinking (`enable_thinking=false` / `--reasoning-budget 0`).
- Ước tính decode Q4_K_M trên 5070 Ti khoảng 80–110 token/s. Đây là số từ các trang ước tính, chưa đo thật.

### Kết quả đo GGUF trên bộ 37 vùng (llama.cpp, 1×T4)

| Bản | File GGUF | CER vùng* | s/vùng (tuần tự) | 37 vùng, 4 slot song song | prefill tok/s | decode tok/s | VRAM (ctx 4×8192) |
|---|---|---|---|---|---|---|---|
| Q8_0 | 8.9 GiB | 7.9% | 1.44 | 20.4 s | 461 | 22 | 10.5 GiB |
| Q6_K | 7.0 GiB | 7.9% | 1.52 | 20.5 s | 437 | 21 | 8.8 GiB |
| Q4_K_M | 5.3 GiB | 8.9% | 1.09 | 16.0 s | 538 | 32 | 7.4 GiB |
| UD-Q4_K_XL | 5.6 GiB | 10.1% | 1.37 | 18.6 s | 464 | 25 | 7.6 GiB |

Nhận xét:
- llama.cpp nhanh hơn transformers trên Kaggle khoảng **10–14 lần** (1.1–1.5 s/vùng so với 15 s). Chạy 4 slot song song còn khoảng **0.45–0.55 s/vùng**.
- Q6_K và Q8_0 cho độ chính xác như nhau. Q4_K_M sai thêm khoảng 1 điểm CER. Bản UD-Q4_K_XL của unsloth không tốt hơn Q4_K_M.
- Chênh lệch giữa fp16 (7.1%) và Q8_0 (7.9%) chủ yếu do một vùng ngắn là `p1_tokens`: llama.cpp đọc thành ký hiệu phiên âm (`[mɛ̃] [ɑ̃]`). Nhiều khả năng do cách llama.cpp tiền xử lý ảnh khác transformers, không phải do quantize.
- **Khuyên dùng Q6_K**: chính xác ngang Q8_0 và tốn 8.8 GiB, nên còn chỗ cho MinerU trên card 16 GB. Nếu cần tiết kiệm VRAM hoặc cần nhanh hơn, dùng Q4_K_M.
- Script chạy: `scripts/bench_gguf.py`. Log: `logs/run_gguf.log`, `logs/gguf_results.md`.
