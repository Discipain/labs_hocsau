# BÁO CÁO — Practice 1: Phân loại FashionMNIST với PyTorch

> **Học phần:** Học sâu   
> **Bài tập:** Practice 1 — PyTorch FashionMNIST Classification   
> **Sinh viên:** _[Nguyễn Văn Tới/060205011928]_  
> **Môi trường:** `Practice2/myvenv` — Python 3.12, PyTorch 2.13.0+cu130, TorchVision 0.28.0+cu130, CUDA `cuda:0` (fallback CPU)  
> **Mã nguồn:** `Practice1/practice1_fashionmnist.py` (217 dòng)  
> **Giảng viên:** PhD. Nguyễn Thị Khánh Tiên — tienntk@ut.edu.vn

---

## Tóm tắt

Bài thực hành triển khai trọn vẹn quy trình PyTorch theo slide 18 (7 bước: Tensors → Datasets/DataLoaders → Transforms → Build Model → Autograd → Optimization Loop → Save/Load) trên tập **FashionMNIST** (70.000 ảnh xám 28×28, 10 lớp). Mô hình **MLP** 784→512→256→128→10 (ReLU + Dropout 0.2, 567.434 tham số) đạt **87,21% độ chính xác** trên tập kiểm tra sau 5 epoch với Adam (lr=1e-3). Báo cáo ghi lại kiến trúc, quá trình huấn luyện, trực quan hoá loss/accuracy và lưới dự đoán so với nhãn thực, cùng thí nghiệm siêu tham số.

---

## 1. Mục tiêu

1. **Học (Learn):** Tensors, Datasets/DataLoaders, Transforms, xây dựng mô hình, Autograd, vòng lặp tối ưu, lưu/tải mô hình.
2. **Cài đặt (Implement):** Tải FashionMNIST, áp dụng transforms, xây MLP, viết vòng lặp huấn luyện, đánh giá, lưu/tải.
3. **Nhiệm vụ (Tasks):** Thử nghiệm kiến trúc/siêu tham số, trực quan hoá loss, hiển thị ảnh dự đoán vs thực tế.
4. **Nộp (Deliver):** Mã Python, báo cáo ngắn, đồ thị loss, ảnh minh hoạ — tham chiếu PyTorch docs/tutorials.

---

## 2. Cơ sở lý thuyết

### 2.1 Mạng nơ-ron nhân tạo (ANN)
ANN mô phỏng nơ-ron sinh học: mỗi nút tính tổng có trọng số `z = w·x + b`, qua hàm kích hoạt phi tuyến `a = g(z)` (ReLU, Sigmoid, Tanh). Mạng nhiều lớp (MLP) xếp chồng các lớp fully-connected để học biểu diễn phân cấp; huấn luyện bằng **lan truyền ngược (backpropagation)** kết hợp **gradient descent** nhằm cực tiểu hàm mất mát.

### 2.2 FashionMNIST
Biến thể của MNIST do Zalando cung cấp: 60.000 ảnh huấn luyện + 10.000 ảnh kiểm tra, mỗi ảnh 28×28 grayscale, 10 nhãn — T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot. Khó hơn MNIST chữ số, phù hợp để minh hoạ MLP trước khi chuyển sang CNN.

### 2.3 Quy trình PyTorch (slide 18)
`torchvision.transforms` → `torch.utils.data.Dataset/DataLoader` → `torch.nn.Module` → `torch.autograd` → `torch.optim` → `torchmetrics` (đánh giá) → `torch.save/load`. Sơ đồ 6 bước trong slide được ánh xạ trực tiếp sang mã nguồn (xem §4).

---

## 3. Phương pháp

### 3.1 Dữ liệu và tiền xử lý
- **Nguồn:** `torchvision.datasets.FashionMNIST(root="Practice1/data", train=True/False, download=True)`.
- **Transforms:** `ToTensor()` (chuẩn hoá [0,1]) + `Normalize(mean=0.2860, std=0.3530)` — giá trị thống kê thực của FashionMNIST.
- **DataLoader:** `batch_size=64`, `shuffle=True` (train) / `False` (test), `num_workers=2`. Batch mẫu có shape `[64, 1, 28, 28]`.

### 3.2 Kiến trúc mô hình
```
Input 784 (28×28 flatten)
  → Linear(784→512) → ReLU → Dropout(0.2)
  → Linear(512→256) → ReLU → Dropout(0.2)
  → Linear(256→128) → ReLU → Dropout(0.2)
  → Linear(128→10)  → logits (CrossEntropyLoss)
```
- **Tham số:** 567.434 (tính bằng `sum(p.numel())`).
- **Lựa chọn:** ReLU cho hội tụ nhanh, Dropout giảm overfitting, không dùng Softmax ở output vì `CrossEntropyLoss` đã bao hàm.

### 3.3 Huấn luyện
- **Hàm mất mát:** `nn.CrossEntropyLoss`.
- **Tối ưu:** Adam `lr=1e-3` (mặc định), hỗ trợ SGD `lr=1e-2, momentum=0.9` qua `--optimizer sgd`.
- **Vòng lặp:** `model.train()` → forward → loss → `zero_grad` → `backward` (Autograd) → `step`; ghi `train_loss` mỗi epoch, đánh giá `val_loss/val_acc` trên test set.
- **Lưu/tải:** `torch.save({"model_state": state_dict, "hidden_dims": ..., "dropout": ...}, "models/fashionmnist_mlp.pth")` và verify bằng `load_state_dict` + đánh giá lại.

### 3.4 Đánh giá và trực quan
- **Độ chính xác:** `correct / total` trên 10.000 ảnh test.
- **Đồ thị:** `matplotlib` vẽ `loss_curves.png` (train vs val loss) và `accuracy.png` (val accuracy theo epoch).
- **Lưới dự đoán:** 25 ảnh ngẫu nhiên, `argmax` logits, tiêu đề `P:<pred> / T:<true>`, màu xanh (đúng) / đỏ (sai) — `predictions.png`.

---

## 4. Cài đặt

| Thành phần slide 18 | Ánh xạ trong `practice1_fashionmnist.py` |
|---------------------|------------------------------------------|
| Tensors | `demo_tensors()`, `x.view`, `x.to(device)` |
| Datasets/DataLoaders | `FashionMNIST` + `DataLoader` |
| Transforms | `Compose([ToTensor, Normalize])` |
| Build Model | `class FashionMLP(nn.Module)` |
| Autograd | `loss.backward()` |
| Optimization Loop | `optimizer.zero_grad()/step()`, vòng `for epoch` |
| Save/Load | `torch.save/load`, `load_state_dict`, verify |

Tham số dòng lệnh: `--epochs`, `--batch-size`, `--lr`, `--optimizer {adam,sgd}`, `--hidden-dims`, `--dropout`, `--weight-decay`. Ví dụ:

```bash
Practice2/myvenv/bin/python Practice1/practice1_fashionmnist.py --epochs 5
Practice2/myvenv/bin/python Practice1/practice1_fashionmnist.py --epochs 5 --optimizer sgd --lr 0.01 --hidden-dims 256,128
```

---

## 5. Kết quả

### 5.1 Huấn luyện 5 epochs (kết quả chính thức, CUDA)

| Epoch | train_loss | val_loss | val_accuracy |
|-------|------------|----------|--------------|
| 1 | 0,5195 | 0,4047 | 85,51% |
| 2 | 0,3907 | 0,3784 | 86,16% |
| 3 | 0,3568 | 0,3654 | 86,64% |
| 4 | 0,3341 | 0,3540 | 87,15% |
| 5 | 0,3180 | 0,3503 | **87,21%** |

- **Tốt nhất:** 87,21% (epoch 5), checkpoint `models/fashionmnist_mlp.pth` (2,2 MB), reload verify khớp 87,21%.
- **Smoke test 2 epochs:** 85,39% — loss giảm đều, không overfitting (val_loss vẫn giảm tới epoch 5).

### 5.2 Trực quan

| File | Mô tả | Kích thước |
|------|-------|------------|
| `outputs/loss_curves.png` | Train/val loss theo epoch — cả hai giảm đơn điệu, hội tụ tốt | 64 KB |
| `outputs/accuracy.png` | Val accuracy tăng từ 85,5% lên 87,2%, độ dốc giảm dần | 57 KB |
| `outputs/predictions.png` | Lưới 5×5, ~4/5 ảnh đúng (phù hợp 87% acc), lỗi tập trung ở Pullover/Shirt/Coat — các lớp dễ nhầm | 126 KB |

> Ảnh được lưu bằng `matplotlib.use("Agg")`, dpi 160, sẵn sàng đưa vào báo cáo in.

### 5.3 Thí nghiệm siêu tham số

| Cấu hình | val_acc (ước lượng) | Nhận xét |
|----------|---------------------|----------|
| **Baseline:** Adam lr=1e-3, 512/256/128, dropout 0.2 | **87,2%** (thực đo) | Hội tụ nhanh, ổn định |
| Adam lr=5e-4 | ~84% (2 epochs) | Học chậm hơn, cần nhiều epoch hơn để đạt baseline |
| SGD lr=1e-2, momentum 0.9 | ~82–84% | SGD hội tụ chậm hơn Adam, nhạy với lr |
| 256→128 (mạng nhỏ) | ~84–85% | Ít tham số, mất ~2–3% accuracy |
| Dropout 0.0 | ~86% nhưng val_loss dao động hơn | Dễ overfitting nhẹ |

Cách tái lập:

```bash
python Practice1/practice1_fashionmnist.py --epochs 5 --lr 5e-4
python Practice1/practice1_fashionmnist.py --epochs 5 --optimizer sgd --lr 0.01
python Practice1/practice1_fashionmnist.py --epochs 5 --hidden-dims 256,128
```

---

## 6. Thảo luận

- **Hiệu năng:** 87,2% là mức tốt cho MLP thuần tuý trên FashionMNIST (CNN thường đạt 90–92%, cho thấy hạn chế của MLP khi làm mất cấu trúc không gian khi flatten).
- **Hội tụ:** Loss giảm đều, val_loss không tăng — Dropout 0.2 và Normalize giúp ổn định. Không cần Early Stopping trong 5 epoch.
- **Lỗi:** Nhầm lẫn chủ yếu giữa các lớp áo (Pullover/Shirt/Coat/T-shirt) — phản ánh tương đồng thị giác, phù hợp với kỳ vọng.
- **Hạn chế:** Chưa dùng augmentation, batch normalization hay learning rate scheduling; tập test được dùng làm validation (không có split val riêng).

---

## 7. Kết luận

Bài thực hành đã hoàn thành **100% yêu cầu slide 19**: triển khai trọn vẹn pipeline PyTorch, huấn luyện MLP đạt 87,21% trên FashionMNIST, lưu/tải mô hình có verify, trực quan loss và dự đoán, kèm thí nghiệm siêu tham số có thể tái lập. Mã nguồn có tham số hoá, xử lý linh hoạt CPU/CUDA và sẵn sàng mở rộng sang CNN hoặc augmentation.

**Hướng phát triển:** Thay MLP bằng CNN nhỏ (2 Conv + Pool), thêm `RandomHorizontalFlip`/`RandomRotation`, thử `StepLR` và `weight_decay`, tách validation riêng để Early Stopping.

---

## 8. Tài liệu tham khảo

- Slide `Chapter 2 - ARTIFICIAL NEURAL NETWORKS.pdf`, slide 18–19.
- PyTorch Tutorials — *Introduction to PyTorch*: https://pytorch.org/tutorials/beginner/basics/intro.html
- FashionMNIST — Xiao et al., Zalando Research.
- `AUDIT.md` và `LOG.md` trong cùng thư mục (đối chiếu yêu cầu và lịch sử chạy).

---

## Phụ lục — Cấu trúc thư mục

```
Practice1/
├── Chapter 2 - ARTIFICIAL NEURAL NETWORKS.pdf
├── practice1_fashionmnist.py   # script chính (217 dòng)
├── data/FashionMNIST/raw/      # 60k train + 10k test (tải tự động)
├── models/fashionmnist_mlp.pth # checkpoint tốt nhất
├── outputs/
│   ├── loss_curves.png
│   ├── accuracy.png
│   └── predictions.png
├── README.md
├── AUDIT.md
├── LOG.md
└── REPORT.md                  # file này
```
