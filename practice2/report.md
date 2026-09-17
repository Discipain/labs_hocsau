# Báo cáo Practice 2 — Hands-on Practice with Pre-trained Neural Network Architectures

**Môn:** Học phần sâu (Deep Learning) — Chapter 3

---

## 1. Mục tiêu

Theo đề bài:
1. Làm quen với việc load và sử dụng mô hình khởi tạo trước (pre-trained) từ `torchvision.models`.
2. Hiểu cách điều chỉnh (adapt) và tinh chỉnh (fine-tune) mô hình pre-trained cho tác vụ cụ thể.
3. Thực hành 3 bài tập mở rộng: so sánh nhiều kiến trúc, điều chỉnh siêu tham số, giám sát bằng TensorBoard.

## 2. Môi trường

| Thành phần | Giá trị |
|------------|---------|
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| Torchvision | 0.28.0 |
| TensorBoard | 2.21.0 |
| Thiết bị huấn luyện | **CPU 16 nhân** (RTX 4050 không khả dụng trong môi trường Linux đang dùng — driver NVIDIA đã cài nhưng GPU không xuất hiện trên PCIe bus; xem Phụ lục D) |
| Dataset | CIFAR-10 — 50.000 ảnh train / 10.000 ảnh test, 10 lớp, 32×32 |

**Trọng số pre-trained:** tải từ `download.pytorch.org` qua API `weights=*.DEFAULT` (torchvision ≥ 0.13).

## 3. Step 2–4 — Load, khám phá và điều chỉnh mô hình

### 3.1 API load weights

`pretrained=True` (như trong slide) đã **lỗi thời**; báo cáo dùng API mới, tương đương về bản chất — cùng bộ trọng số ImageNet:

```python
from torchvision import models
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
```

### 3.3 Kiến trúc và tham số (Step 3)

Bảng sinh ra từ `practice2/explore_model.py` (kiến trúc đầy đủ: `practice2/logs/arch_*.txt`):

| Model | Tổng params (ImageNet 1000 lớp) | Lớp phân loại | Params lớp phân loại | Sau khi thay head (10 lớp) |
|-------|--------------------------------:|---------------|---------------------:|---------------------------:|
| ResNet18 | 11.689.512 | `model.fc` | 513.000 | 11.181.642 |
| VGG16 | 138.357.544 | `model.classifier[6]` | 4.097.000 | 134.301.514 |
| DenseNet121 | 7.978.856 | `model.classifier` | 1.025.000 | 6.964.106 |

**Nhận xét:** VGG16 nặng gấp ~12 lần ResNet18, chủ yếu nằm ở 2 layer fully-connected 4096 nơ-ron; DenseNet121 nhẹ nhất (121 lớp nhưng dùng dense block + bottleneck nên ít tham số).

### 3.4 Điều chỉnh cho CIFAR-10 (Step 4) — hai chiến lược

```python
# Transfer learning: đông cứng toàn bộ backbone, thay head
for p in model.parameters(): p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)   # head mới tự requires_grad=True

# Fine-tuning: mở khóa thêm block cuối
for p in model.layer4.parameters(): p.requires_grad = True
```

| Chiến lược | Params được học (ResNet18) | Learning rate backbone |
|------------|---------------------------:|------------------------|
| Transfer (freeze) | 5.130 (0,05%) | — (không cập nhật) |
| Fine-tune layer4 | 8.398.858 (75,11%) | lr/10 (param groups) |

Vì sao fine-tune cần lr nhỏ hơn cho backbone đã pre-trained: tránh phá hủy đặc trưng tốt đã học trên ImageNet ("catastrophic forgetting").

## 4. Step 5 — Chuẩn bị dữ liệu

```python
train_tf = transforms.Compose([
    transforms.Resize(224),                    # pre-trained yêu cầu đầu vào 224×224
    transforms.RandomHorizontalFlip(),         # augmentation chống overfitting
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])  # chuẩn ImageNet
])
```

Hai điểm bắt buộc khi dùng pre-trained: **Resize về đúng đầu vào lúc train trên ImageNet**, và **Normalize bằng đúng mean/std ImageNet** — sai một trong hai là trọng số pre-trained mất ý nghĩa.

## 5. Step 6 — Vòng lặp huấn luyện

6 bước chuẩn Chapter 3 (`practice2/train.py::train_one_epoch`):

```python
data, target = data.to(device), target.to(device)  # 1. chuyển thiết bị
optimizer.zero_grad()                              # 2. xóa gradient cũ (PyTorch cộng dồn!)
output = model(data)                               # 3. forward
loss = criterion(output, target)                   # 4. CrossEntropyLoss
loss.backward()                                    # 5. backward (backpropagation)
optimizer.step()                                   # 6. cập nhật weights
```

Chi tiết: `filter(lambda p: p.requires_grad, ...)` ở transfer; param groups `[backbone lr/10, head lr]` ở fine-tune; `weight_decay=1e-4` (L2).

## 6. Step 7 — Đánh giá

`model.eval()` + `@torch.no_grad()` + per-class accuracy + confusion matrix (xem ảnh `.png` kèm theo).

---

## 7. Exercise 1 — So sánh các kiến trúc khởi tạo trước

*Cùng cấu hình: transfer, Adam lr=1e-3, batch=32, 6 epochs (VGG16: 4).*

| Model | Thời gian/epoch | Val Acc cuối | Nhận xét |
|-------|-----------------|--------------|----------|
| ResNet18 | *<điền>* | *<điền>* | |
| DenseNet121 | *<điền>* | *<điền>* | |
| VGG16 | *<điền>* | *<điền>* | |

Ảnh đường cong: `logs/ex1_*_curves.png` — Confusion: `logs/ex1_*_cm.png`

**Kết luận (điền sau khi có số):** model nào thắng, có khớp với lý thuyết về kiến trúc (residual connection, density) không.

## 8. Exercise 2 — Điều chỉnh siêu tham số

*Cơ sở: ResNet18 transfer. Bốn hàng dưới đây tách từ các run `ex2_*` và `ex1_resnet18_transfer`.*

| Run | Optimizer | lr | batch_size | Val Acc cuối | Thời gian |
|-----|-----------|-----|------------|--------------|-----------|
| ex1_resnet18_transfer (gốc) | Adam | 1e-3 | 32 | *<điền>* | |
| ex2_lr1e-4 | Adam | **1e-4** | 32 | *<điền>* | |
| ex2_bs64 | Adam | 1e-3 | **64** | *<điền>* | |
| ex2_sgd | **SGD m=0.9** | 1e-2 | 32 | *<điền>* | |

**Kết luận (điền):** ảnh hưởng của từng siêu tham số tới tốc độ hội tụ và accuracy; lr nhỏ → chậm hơn bao nhiêu; batch to → ít step hơn nhưng step lâu hơn...

## 9. Transfer vs Fine-tune

| | Transfer | Fine-tune (layer4) |
|---|---|---|
| Params học | 5.130 | 8.398.858 |
| Val Acc | *<điền ex1_resnet18>* | *<điền ex_ft_resnet18>* |
| Thời gian | *<điền>* | *<điền>* |

## 10. Exercise 3 — TensorBoard

```bash
myvenv/bin/tensorboard --logdir practice2/runs --port 6006
```
Mở `http://localhost:6006`. Tất cả các run chồng lên nhau trên cùng biểu đồ (`Loss/*`, `Accuracy/*`, `Epoch/*`); tab **Graphs** hiển thị kiến trúc model; ảnh chụp đính kèm *(chèn sau khi xem)*.

## 11. Kết luận

*(điền: 3–5 dòng — pre-trained + transfer cho kết quả tốt nhất với dữ liệu nhỏ/ít epoch; fine-tune cải thiện thêm khi cần; kiến trúc residual hiệu quả/thời gian chạy so với VGG; bài học từ lr, batch size)*

---

## Phụ lục A — Cách chạy lại toàn bộ

```bash
cd Practice2
myvenv/bin/python practice2/explore_model.py           # Step 2-4: kiến trúc + params
myvenv/bin/python practice2/train.py --model resnet18 --mode transfer --epochs 6 \
    --lr 1e-3 --batch-size 32 --run-name demo          # một run lẻ
bash practice2/run_experiments.sh                      # cả 7 experiments (chạy ~5-6 tiếng)
```

## Phụ lục B — Bảng kết quả thô

Mỗi run có log đầy đủ tại `practice2/logs/<run>.log` (đường epoch/loss/acc, per-class).

## Phụ lục C — Câu hỏi phản biện có thể bị hỏi

1. *Vì sao không train từ đầu mà dùng pre-trained?* — CIFAR-10 quá nhỏ so với ImageNet để học đặc trưng từ số 0 trong vài epoch; transfer hội tụ nhanh hơn nhiều (đã chứng minh ở Lab thực hành: freeze model random chỉ đạt ~36% vs pre-trained ~70%+ cùng điều kiện).
2. *Vì sao fine-tune phải dùng lr nhỏ hơn cho backbone?*
3. *Resize(224) từ ảnh 32×32 có gì đáng chê? Vì sao không dùng model thiết kế cho CIFAR?* — phóng to làm mất chi tiết; ResNet CIFAR gốc dùng conv 3×3 stride 1 và input 32×32 sẽ nhanh và hợp hơn, nhưng đề bài yêu cầu thực hành pre-trained.
4. *Batch size ảnh hưởng thế nào tới gradient và tốc độ?*
5. *Overfitting xuất hiện chưa, nhìn ở đâu?* — khoảng hở train/val trên đường cong.

## Phụ lục D — Ghi chú về GPU

RTX 4050 của máy không được hệ điều hành nhận (không xuất hiện trên PCIe bus; chỉ iGPU AMD Radeon 680M hoạt động — có thể do chế độ GPU Eco trong BIOS/Armoury Crate). Toàn bộ thí nghiệm chạy CPU; thời gian mỗi run ghi trong bảng là trên CPU 16 nhân.
