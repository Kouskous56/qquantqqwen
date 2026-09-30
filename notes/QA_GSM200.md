# Q&A GSM-200 / GSM-400 — vì sao con số "s20 hơn dense" phải bị bác

Ngày: 2026-09-30. Đây là hồ sơ giải thích một kết quả **không kết luận được**,
để tránh nó bị tái phát hiện và báo cáo như phát hiện.

## Q1. Có phải Wanda s20 tốt hơn dense không?

Không. Đây là nhiễu, và bản thân dữ liệu đã tự bác bỏ.

## Q2. Chuỗi dữ liệu diễn ra thế nào?

| bước | n | dense | s20-C4 | $p$ (McNemar exact) |
|---|---|---|---|---|
| Block A (câu 0–199) | 200 | 116 | **128** | **0.0428** |
| Block B (câu 200–399) | 200 | **122** | 122 | 1.0000 |
| Gộp 400 | 400 | 238 | 250 | 0.1263 |

Block A cho $p=0.043$ — trông như phát hiện. Block B cho $p=1.000$ — **dấu hiệu
đảo chiều hoàn toàn**. Gộp lại: không có ý nghĩa.

Độ rộng ước lượng: ở 400 câu, cần khoảng **800 câu** mới đủ lực cho một hiệu ứng
3pp với $p<0.05$. Ta có 400, thiếu khoảng một nửa.

## Q3. Có phải tôi "chạy replicate quá mức" không?

Không — đây là **garden-of-forking-paths điển hình**. Ta chạy block A, thấy
$p=0.043$, rồi mới chạy block B. Vì vậy con số gộp 400 là **exploratory,
post-hoc**, không phải confirmatory pre-registered. Bất kỳ ai đọc con số
này phải biết điều đó.

Bài học thực tế: nếu ta chỉ chạy block A rồi viết báo cáo, ta đã công bố một
kết quả sai. Chi phí của replicate là 11 phút; chi phí của việc không
replicate là mất uy tín.

## Q4. Vậy còn hiện tượng nào đáng giữ không?

Hai hiện tượng khác, **đều đã được kiểm chứng và vẫn đúng**:

**a) Recovery là domain-bound** (mạnh nhất). Cùng một cặp model, hai kiến trúc,
hai cơ chế recovery khác nhau, cùng một bất đối xứng:

| | toán | tri thức |
|---|---|---|
| Qwen2.5-0.5B + LoRA masked | GSM 4 → 14 (+10) | generic 9 → 11 (+2) |
| Qwen3-4B + EoRA rank-128 | GSM 33 → 35 (+2) | suite typed 43 → 42 (−1) |

**b) Prune + Q4 đảo chiều so với FP16** (chưa phân giải, xem Q5).

## Q5. Hiện tượng Q4 có thật không?

Chưa biết, và có **hai lý do nghi ngờ nghiêm trọng** — một trong hai là lỗi
của chính tôi:

**Lý do 1 — confound thủ công.** 5 artifact Q4 dùng **hai quy trình lượng tử
khác nhau**. Đọc metadata GGUF:

| artifact | imatrix |
|---|---|
| `Qwen3-4B-s20-c4-Q4KM.gguf` (28/9) | có |
| `Qwen3-4B-s20-c4-eora128-Q4KM.gguf` (28/9) | có |
| `DENSE-Q4KM.gguf` (29/9) | **không** |
| `S20MIX-Q4KM.gguf` (29/9) | **không** |
| `S30MIX-Q4KM.gguf` (29/9) | **không** |

Đúng cặp cho ra "s20 hơn dense" là dense (thuần) vs s20-C4 (imatrix). Mọi phát
hiện Q4 trước 30/9 vì vậy **không dùng được** cho tới khi quantize lại đồng
quy trình.

**Lý do 2 — FP16 và Q4 không cùng thang.** Trên **cùng 50 câu đầu**, cùng
model (`Qwen3-4B-FP16` đã xác nhận là Instruct-2507 qua model card), cùng
prompt, cùng greedy:

| | GSM 50 câu đầu |
|---|---|
| dense FP16 (transformers) | 36 (72.0%) |
| dense Q4 (Ollama, thuần) | 29 (58.0%) |

Chênh 14pp. Với Q4_K_M có imatrix, chênh này gần như không thể xảy ra.

## Q6. Giả thuyết "prune làm giảm sai số lượng tử" còn đáng thử không?

**Đã bị bác bằng đo đạc trực tiếp.** Ý tưởng: trong K-quant, nhiều zero →
ít giá trị phân biệt → step nhỏ hơn → sai số giảm.

Đo `range = max − min` cho từng sub-block 32 phần tử (đúng kích thước sub-block
của Q4_K), dense so với prune-20%:

| tensor | range dense | range pruned | chênh |
|---|---|---|---|
| L0 q_proj | 0.09274 | 0.09274 | 0.000% |
| L17 o_proj | 0.08767 | 0.08767 | −0.000% |
| L35 down_proj | 0.09392 | 0.09387 | −0.048% |

**Chênh lệch trung bình 0.005%, lớn nhất 0.048%.** Zero: dense 0.000%,
pruned 20.000% (xác nhận mask chính xác).

Kết luận: trong K-quant của llama.cpp, scale đặt theo `iscale = nmax/(max−min)`.
Wanda cắt 20% trọng số có $|w|$ nhỏ nhất — **không chạm max lẫn min**. Nên
`max−min` bất biến, scale bất biến, và cơ chế "nhiều zero → bớt lỗi" **không
tồn tại**. Tệ hơn: trong K-quant có affine offset `min`, nên 0 **không** được
mã hoá miễn phí (tái tạo qua $d\cdot sc\cdot L + \min$, lệch tới step/2).
Chỉ Q4_0 legacy (max-abs thuần) mới cho zero chính xác.

Người đọc source `ggml/src/ggml-quants.c` đã dự đoán trước kết quả này; đo đạc
chỉ xác nhận.

## Q7. Vậy nên làm gì với block Q4?

**Không viết gì vào báo cáo cho tới khi quantize lại.** Thứ tự:

1. Quantize cả 5 arm cùng một quy trình (dùng `Qwen3-imatrix.dat` sẵn có)
2. Chạy lại MMLU-200 và GSM-400
3. Xem "đảo chiều" còn không
4. Nếu còn → phát hiện thật, viết vào. Nếu mất → ghi thẳng là artifact quy
   trình lượng tử; đây là negative result có giá trị và **không tốn** thêm gì.

## Q8. Còn cách nào đo đúng hơn không?

Có, và rẻ hơn nhiều so với thêm benchmark:

- **Đo trực tiếp sai số tái dựng** $\|W - Q(W)\|_F/\|W\|_F$ cho Q8_0/Q4_K_M/Q3_K
  dense vs pruned, kèm phân rã bao nhiêu % lỗi đến từ nhóm trọng số đã bị prune.
  Đây là đại lượng đang giả định, đo trực tiếp thay vì suy ra.
- **Q4_0** (max-abs thuần) là format thuận lợi nhất cho giả thuyết "zero miễn
  phí". Nếu không thấy hiệu ứng ở đó, giả thuyết chết dứt khoát.
- **WikiText2 perplexity** thay vì GSM8K: nhiễu thấp hơn nhiều, chạy vài phút.

Không nên chạy thêm GSM8K ở bước này — nhiễu quá lớn so với hiệu ứng cần đo.

## Q9. Tại sao GSM8K lại nhiễu lớn như vậy?

Vì Qwen3 sinh CoT dài. Ở temperature 0, thay đổi logit nhỏ đủ đảo một token,
 và độ chính xác **không đơn điệu** theo "chất lượng mô hình". Cùng một model
với perturbation nhỏ có thể đổi cả chuỗi suy luận. Đây là công cụ đo sai cho
giả thuyết cơ chế; phải dùng perplexity hoặc probe token đầu.
