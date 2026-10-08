# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Anh Tuấn
**Khoá:** A20-K4
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Cấu hình lấy từ `lab22/config.py`; các kết quả NB0 lấy từ output notebook.
> Chỉ số NB2 cần đối chiếu lại với `data/pref/stats.json` của phiên Colab; các chỉ số NB3–NB4
> chỉ được điền sau khi có file kết quả, không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 15 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch` |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (vi) · 800 cặp huấn luyện / 100 cặp held-out (không trùng prompt) |
| Chosen / rejected median (NB2) | 94 / 86 token |
| Chosen dài hơn rejected (NB2) | 65,9% |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1 (cấu hình T4) |
| Giám khảo | _<rm:tên-mô-hình hoặc nhà-cung-cấp:tên-mô-hình; sanity accuracy>_ |
| Chi phí | _<0 đồng (Colab miễn phí) / ...>_ |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | _<...>_ |
| VRAM cao nhất | _<...>_ |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | _<...>_ |
| Độ chính xác reward trên held-out | _<...>_ |
| Margin trên held-out | _<...>_ |
| Chẩn đoán tự động (`diagnosis`) | _<INTENDED / LIKELIHOOD DISPLACEMENT / FAILURE / AMBIGUOUS>_ |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | _<... → ... ký tự>_ |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

_Mô tả riêng `rewards/chosen` và `rewards/rejected` trên **train và held-out**. Chosen tăng hay giảm?
Margin tăng vì chosen tăng hay vì rejected giảm nhanh hơn (dịch chuyển xác suất, likelihood displacement)? Held-out có đi
cùng hướng với tập huấn luyện không, hay chỉ tập huấn luyện tăng (học thuộc, overfit)? Chẩn đoán tự động có khớp với điều bạn
thấy không?_

Ở NB0, khi mô hình đang học trùng mô hình tham chiếu, reward của cả `chosen` và `rejected` đều bằng 0, nên loss là 0,6931 (`log 2`). Hai kịch bản A và B cùng có margin bằng 2 và DPO loss bằng 0,127. Trong kịch bản B, log-xác suất của `chosen` giảm 3 nat nhưng của `rejected` giảm 5 nat; hiệu giữa hai thay đổi vẫn là 2. Vì DPO tối ưu hiệu này, loss có thể giảm dù câu trả lời được chọn cũng mất xác suất: đó là *likelihood displacement*. Khi thêm NLL của `chosen`, RPO phạt kịch bản B nặng hơn A (2,427 so với 2,027). DPO dùng tổng log-xác suất trên toàn câu nên độ dài còn ảnh hưởng độ lớn tín hiệu; SimPO và ORPO dùng log-xác suất trung bình theo token để giảm thiên lệch này. Cần đối chiếu riêng hai đường `chosen`/`rejected` trên train và held-out ở NB3 trước khi kết luận mô hình thực tế thuộc chẩn đoán nào.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | | | | | | | |
| hữu ích — helpfulness (4) | | | | | | | |
| an toàn — safety (4) | | | | | | | |

Giám khảo: ______ · sanity accuracy: ______ · `score_length_spearman` (reward model) hoặc độ nhất quán khi đổi chỗ A/B — position consistency (giám khảo API): ______

_Khoảng tin cậy có chứa 0.5 không? Giám khảo có đáng tin trên tiếng Việt không (xem bộ cặp kiểm tra sanity)? DPO thắng vì câu trả lời tốt
hơn hay vì dài hơn? Hai reward model trong hội đồng (`per_judge`) có cho win rate gần nhau không? Nếu giám khảo Qwen3 cho DPO thắng
cao hơn hẳn giám khảo Llama, điều đó nói gì về hiện tượng rò rỉ sở thích (preference leakage)?
Chọn 2 ví dụ cụ thể (1 câu về độ hữu ích, 1 câu về an toàn) và giải thích._

_Trả lời ở đây._

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

Nếu không chạy β-sweep, tôi dự đoán β = 0,05 cho phép policy lệch khỏi reference SFT nhiều hơn, nên margin held-out có thể lớn hơn nhưng rủi ro dịch chuyển xác suất và học thuộc cũng tăng. Với β = 0,5, tôi kỳ vọng thay đổi bảo thủ hơn và margin nhỏ hơn; tuy nhiên số bước và tốc độ học cố định có thể khiến xu hướng thực tế khác dự đoán. Đây chỉ là giả thuyết, chưa phải kết quả đo.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

_Trả lời ở đây._

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

_(Tuỳ chọn, 1–3 câu)_
