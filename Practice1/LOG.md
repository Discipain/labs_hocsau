# LOG — Practice 1: FashionMNIST Classification

> **Thư mục:** `Practice1/`  
> **Script chính:** `practice1_fashionmnist.py` (217 dòng, 9.1 KB)  
> **Môi trường:** `Practice2/myvenv` — Python 3.12, torch 2.13.0+cu130, torchvision 0.28.0+cu130, CUDA `cuda:0`

---

## Lịch sử chạy

### 2026-09-17 — Cài đặt & chuẩn bị
```bash
Practice2/myvenv/bin/pip install transformers datasets evaluate accelerate scikit-learn matplotlib
# -> Successfully installed transformers-5.17.0, datasets-5.0.1, evaluate-0.4.6, accelerate-1.15.0, ...
mkdir -p Practice1/models Practice1/outputs
```

### 2026-09-17 13:2x — Smoke test 2 epochs
```bash
Practice2/myvenv/bin/python Practice1/practice1_fashionmnist.py --epochs 2 --batch-size 64
```
- Tải FashionMNIST lần đầu: 26.4M train-images, 4.42M train-labels, 29.5k t10k… (download OK)
- Device: `cuda`
- Model: 784→512→256→128→10, Dropout 0.2, 567.434 params, Adam lr=1e-3
- Epoch 1: train_loss=0.5182 val_loss=0.4105 val_acc=85.18%
- Epoch 2: train_loss=0.3891 val_loss=0.4020 val_acc=85.39%
- Best val_acc 85.39% — Saved `models/fashionmnist_mlp.pth` (2.2 MB), reload verify 85.39% khớp
- Saved `outputs/loss_curves.png` (64 KB), `outputs/accuracy.png` (57 KB), `outputs/predictions.png` (126 KB)
- **Kết quả:** PASS

### 2026-09-17 13:24 — Full run 5 epochs (kết quả chính thức)
```bash
Practice2/myvenv/bin/python Practice1/practice1_fashionmnist.py --epochs 5 --batch-size 64
```
| Epoch | train_loss | val_loss | val_acc |
|-------|------------|----------|---------|
| 1 | 0.5195 | 0.4047 | 85.51% |
| 2 | 0.3907 | 0.3784 | 86.16% |
| 3 | 0.3568 | 0.3654 | 86.64% |
| 4 | 0.3341 | 0.3540 | 87.15% |
| 5 | 0.3180 | 0.3503 | 87.21% |
- Best val_acc **87.21%** — checkpoint ghi đè `models/fashionmnist_mlp.pth`, reload verify 87.21%
- 3 ảnh outputs được ghi đè với dữ liệu 5 epochs
- **Kết quả:** PASS — dùng làm số liệu chính trong README

### 2026-09-17 13:27 — Tạo README
- `README.md` ghi bảng 5 epochs, kiến trúc model, cách chạy, gợi ý thí nghiệm hyperparams

---

## Lệnh hữu ích

```bash
# Chạy mặc định 5 epochs
Practice2/myvenv/bin/python Practice1/practice1_fashionmnist.py --epochs 5

# Thí nghiệm hyperparams
Practice2/myvenv/bin/python Practice1/practice1_fashionmnist.py --epochs 5 --lr 5e-4
Practice2/myvenv/bin/python Practice1/practice1_fashionmnist.py --epochs 5 --optimizer sgd --lr 0.01
Practice2/myvenv/bin/python Practice1/practice1_fashionmnist.py --epochs 5 --hidden-dims 256,128

# Kiểm tra outputs
ls -lh Practice1/models/ Practice1/outputs/
```

## Ghi chú
- `data/FashionMNIST/raw/` đã có sẵn sau lần đầu download, lần sau chạy không cần tải lại.
- `num_workers=2` cho DataLoader — ổn trên Linux, nếu lỗi trên Windows/Mac đổi về 0.
- Loss giảm đều, không có dấu hiệu overfitting nặng (val_loss vẫn giảm tới epoch 5).
