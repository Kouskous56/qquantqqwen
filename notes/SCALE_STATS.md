# Bằng chứng cho cách đọc mới: claim quy mô (V1, Qwen2.5)

Ngày: 2026-09-30. Nguồn duy nhất: `qwen_release/manifests/RUN_V33_RESCORE.json`
(scorer typed V3.3, `reproduce/scoring.py`).

## 1. Ba cách đọc cùng một dữ liệu

| scale | dense s0 | pruned s30 | **Δ điểm (pp)** | $n_{11}$ | $n_{10}$ mất | $n_{01}$ thắng | $n_{00}$ |
|---|---|---|---|---|---|---|---|
| 0.5B | 18 | 11 | **−14.0** | 8 | 10 | 3 | 29 |
| 1.5B | 40 | 33 | **−14.0** | 33 | 7 | 0 | 10 |
| 3B | 42 | 40 | **−4.0** | 39 | 3 | 1 | 7 |

Ba đại lượng khác nhau, cùng bảng gốc:

- **Điểm tuyệt đối** (pp): 14.0, 14.0, 4.0 — đây là số nên dùng làm chính.
- **Tỷ lệ mất trên tập at-risk** = $n_{10}/(n_{11}+n_{10})$:
  55.6% (CI 30.8–78.5), 17.5% (CI 7.3–32.8), 7.1% (CI 1.5–19.5). CI = Clopper–Pearson.
- **Tỷ lệ giữ** = $(n_{11}+n_{01})/(n_{11}+n_{10})$: 61.1%, 82.5%, 95.2%.

## 2. Vì sao tỷ lệ giữ gây hiểu nhầm

0.5B và 1.5B mất **đúng bằng nhau 14 điểm** (7/50 câu). Tỷ lệ giữ trông "tăng
đơn điệu 61→82.5→95%" nhưng chỉ vì chia cho ba mẫu số khác nhau (18, 40, 42).

Gọi đây là "hiệu ứng trần" là **sai tên**. Retention đã là tỉ lệ hại có điều
kiện, vô hướng theo trần: retention = 1 − (nhiệt)/(số câu dense đúng). Confound
thật là **tương đối so với tuyệt đối**, không phải trần.

## 3. Kiểm định xu hướng

Ba McNemar rời rạc (p = 0.0923, 0.0156, 0.625) **không phải** kiểm định xu
hướng. Cách đúng là kiểm định trên tỷ lệ mất/at-risk:

| so sánh | Fisher exact p |
|---|---|
| 0.5B vs 1.5B (10/18 vs 7/40) | **0.0053** |
| 1.5B vs 3B (7/40 vs 3/42) | 0.189 |
| 0.5B vs 3B (10/18 vs 3/42) | 0.0001 |
| **Cochran–Armitage trend 3 scale** | χ² = 17.6, **p = 0.00003** |

Kết luận: xu hướng **có ý nghĩa**, nhưng **bước có ý nghĩa là 0.5B→1.5B**;
bước 1.5B→3B chưa đủ dữ liệu.

Cũng lưu ý McNemar bỏ qua các cặp đồng thuận, nên nó nói *không* gì về bất
đối xứng thắng/mất — mà bất đối xứng đó mới là thứ phân biệt nhiễu với suy
thoái hệ thống.

## 4. 0.5B là nhiễu hai chiều, không phải đo năng lực

Kiểm lại bằng scorer typed trên raw:

- **3 câu pruned thắng lại** (dense sai, pruned đúng): h20 ($\log_{10}1000=3$),
  h28 ($i^2=-1$), o38
- **10 câu mất**, trong đó phần lớn là **số học cơ bản hỏng**: 12×6 → 18,
  5×5 → 10, 3×4 → 7, 2×4 → 4

Đối chiếu: 1.5B thắng 0 câu, 3B thắng 1 câu (b13). 0.5B là scale duy nhất có
thắng ròng đáng kể.

→ Tỷ lệ giữ 61.1% ở 0.5B đo **độ lớn nhiễu hai chiều**, không đo khả năng
giữ năng lực. **Không được so 61.1% với 82.5% như hai phép đo cùng loại.**

## 5. Vùng đo nằm ngoài tiền lệ

Toàn bộ số liệu tham chiếu "model lớn prune an toàn hơn" đến từ 7B–70B.
Tiêu chí Wanda dựa vào hiện tượng **outlier feature**; LLM.int8() định vị
hiện tượng này xuất hiện **đột ngất quanh 6–6.7B tham số**.

Qwen2.5 0.5B / 1.5B / 3B đều **dưới ngưỡng**. Suy luận từ 7B+ xuống dưới 4B
là không có cơ sở cơ chế. Kết quả này vì vậy **không phải xác nhận** của tiền
lệ, mà là dữ liệu ở vùng chưa ai đo.

## 6. Cách viết đúng

> Ở cùng mức cắt tỉa 30% (Wanda), tổn thất tuyệt đối không giảm đều theo
> quy mô: 0.5B và 1.5B cùng mất 14.0 điểm, 3B mất 4.0. Tỷ lệ mất trên tập
> at-risk giảm 55.6% → 17.5% → 7.1% và xu hướng này có ý nghĩa thống kê
> (Cochran–Armitage p = 0.00003), nhưng bước 1.5B→3B không đạt ý nghĩa khi
> kiểm tra theo cặp (p = 0.189). Ở 0.5B, prune vừa làm hỏng số học cơ bản vừa
> sửa vài lỗi kiến thức rời, nên tỷ lệ giữ ở scale này không cùng loại đại
> lượng với hai scale còn lại.

## 7. Giới hạn chưa giải quyết

- Suite-50 có 50 câu, **không thể mở rộng** — claim này về nguyên tắc không
  cứu được trên metric này. Lối thoát duy nhất là metric n ≥ 400 (GSM-400,
  MMLU-200), và chúng chỉ mới được áp cho Qwen3, chưa áp cho Qwen2.5.
- Muốn kiểm định xu hướng chắc hơn ở 0.5B–1.5B, cần thêm điểm sparsity
  (s10, s50) để tách hiệu ứng quy mô khỏi hiệu ứng ngưỡng tổn thất.
- `dense 0.5B = 36%` trên bộ 50 câu là ngưỡng thấp. Không phải "dưới ngẫu
  nhiên" theo nghĩa MCQ (bộ này là khớp giá trị, không phải chọn chữ cái), nhưng
  vẫn nên kiểm xem model nhỏ có đang bị đo sai thứ (độ nhạy định dạng đầu ra,
  lỗi trích xuất số) chứ không phải năng lực.

## Tái lập

```powershell
python D:\qwen\scripts\scale_stats.py
```

Script đọc thẳng `RUN_V33_RESCORE.json`, dùng `reproduce/scoring.py`, và có
assertion kiểm tra $n_{11}+n_{01}=s_{30}$, $n_{11}+n_{10}=s_{0}$, và
$p$ khớp manifest.
