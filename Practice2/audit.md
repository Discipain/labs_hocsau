# Audit — Tình trạng Practice 2 (ngày 11/09/2026, 13:38)

> File này ghi lại **trạng thái thực tế** của thư mục `practice2/` trước khi tiếp tục — để giảng viên/người kiểm tra biết cái gì đã xong, cái gì còn dở, và vì sao.

## 1. Tổng quan thư mục `practice2/`

| File/Thư mục | Mục đích | Trạng thái |
|---|---|---|
| `train.py` | Script huấn luyện (Steps 2,4,5,6,7) — tham số hóa model/mode/lr/batch, tự xuất PNG | ✅ Đã viết, đã sanity-check OK |
| `explore_model.py` | In kiến trúc + đếm params 3 model (Step 3) | ✅ Đã chạy, ra 3 file `logs/arch_*.txt` |
| `run_experiments.sh` | Chạy tuần tự 7 experiments (3 kiến trúc + finetune + 3 sweep hyperparam) | ✅ Đã viết; đã khởi chạy 03:29 |
| `report.md` | Khung báo cáo nộp giảng viên (có phần câu hỏi phản biện) | ✅ Khung sẵn, chờ điền số |
| `watch.sh` | Giám sát tự động: cập nhật `STATUS.txt` mỗi 60s + ghi `progress.log` | ✅ Đang chạy nền (PID 12626) |
| `logs/` | Chứa kiến trúc, master log, status, progress | ⚠️ Chưa đủ experiment |
| `checkpoints/` | `.pth` mỗi run | ⚠️ Chưa có (do chưa xong epoch) |
| `runs/` | Event TensorBoard | ⚠️ Chưa có run hoàn chỉnh |

## 2. Kết quả đã có

### Explore (Step 2+3+4) — hoàn thành

Nguồn: `practice2/explore_model.py` + `practice2/logs/arch_*.txt`

| Model | Tổng params (ImageNet 1000 lớp) | Lớp head | Params head | Sau khi thay head (10 lớp) |
|---|---|---|---|---|
| ResNet18 | 11.689.512 | `fc` | 513.000 | 11.181.642 |
| VGG16 | 138.357.544 | `classifier[6]` | 4.097.000 | 134.301.514 |
| DenseNet121 | 7.978.856 | `classifier` | 1.025.000 | 6.964.106 |

Chiến lược Step 4:
- **Transfer (freeze)**: chỉ head học → 5.130 params (0,05% ResNet18).
- **Fine-tune (mở layer4)**: head + layer4 → 8.398.858 params (75,11%).

### Training — dở dang

| Thời điểm | Tiến độ | Số liệu mới nhất |
|---|---|---|
| 03:29 | Bắt đầu `ex1_resnet18_transfer` (ResNet18, transfer, Adam 1e-3, bs32, 6 epochs) | — |
| 03:43 | Epoch 1/6 | val_acc **79,6%** (`train_acc 73,4%`) — weights pre-trained đã phát huy (so với 36% khi weights=None ở Lab 07) |
| 03:51 | Epoch 3/6 (1929s) | val_acc **79,8%** — gần như đi ngang, không overfitting |
| 04:08 | Epoch 3/6 (ghi 9791s — timestamp lỗi do chạy qua 2 ngày?) | vẫn 79,8% |
| 13:38 | Kiểm tra thực tế | **Không còn process `run_experiments.sh`** — hàng đợi đã dừng đột ngột, `master.log` chỉ 21 dòng, chưa có dòng `DONE` nào, chưa có checkpoint nào |

**Nhận định:** hàng đợi dừng sau khoảng 30–40 phút (có thể do session Claude bị gián đoạn, hoặc người dùng tắt terminal). `watch.sh` vẫn chạy nhưng không thấy process cha.

## 3. Vấn đề / rủi ro

1. **7 experiments chưa hoàn thành** — chỉ 1 run chạy được ~3/6 epoch; 6 run còn lại chưa bắt đầu.
2. **Không có checkpoint/PNG/TensorBoard run nào** để điền vào `report.md` — báo cáo vẫn ở dạng khung.
3. **Timestamp epoch 4 bất thường (9791s)** — nghi do log ghi đè hoặc lệch epoch đầu do sanity trước đó; cần chạy lại sạch để có log tin cậy.
4. **GPU RTX 4050 không khả dụng** — toàn bộ chạy CPU, mỗi epoch ~10 phút, 7 run × 6 epoch ≈ 5–6 tiếng; cần để máy chạy liên tục.

## 4. Đề xuất khắc phục

- [ ] Khởi động lại hàng đợi: `nohup bash practice2/run_experiments.sh > practice2/logs/master.log 2>&1 &` (ghi đè log cũ để tránh lẫn lộn), hoặc chạy từng run riêng rẽ nếu không muốn mất hết khi gián đoạn.
- [ ] Đảm bảo `watch.sh` được khởi lại cùng (`nohup bash practice2/watch.sh >/dev/null 2>&1 &`).
- [ ] Sau khi xong: chạy `practice2/collect_results.sh` (hoặc đọc `practice2/logs/master.log`) để điền bảng số vào `report.md` (Exercises 1, 2, Transfer vs Fine-tune).
- [ ] Giữ lại file này (`audit.md`) để giảng viên thấy quá trình làm việc có kiểm chứng, không bịa số.

## 5. Cách kiểm tra nhanh sau khi chạy lại

```bash
cat practice2/logs/STATUS.txt          # ảnh chụp hiện tại
tail -f practice2/logs/progress.log    # vết theo thời gian
grep -E "Epoch|START|DONE" practice2/logs/master.log | tail
ls practice2/checkpoints/ practice2/logs/*.png
myvenv/bin/tensorboard --logdir practice2/runs --port 6006
```

---
*Người lập: Claude Code — kiểm tra thực tế lúc 13:38, 11/09/2026.*
