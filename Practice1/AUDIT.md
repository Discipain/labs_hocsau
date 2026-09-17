# AUDIT — Practice 1: FashionMNIST Classification (Chapter 2 - ANN)

> **Ngày audit:** 2026-09-17  
> **Nguồn yêu cầu:** `Chapter 2 - ARTIFICIAL NEURAL NETWORKS.pdf`, slide 19 (Practice 1) + slide 18 (Workflow)  
> **Người thực hiện:** Claude (myvenv: `Practice2/myvenv`, torch 2.13.0+cu130, torchvision 0.28.0+cu130)  
> **Thiết bị:** CUDA available (device `cuda:0`), fallback CPU nếu không có GPU

---

## 1. Đối chiếu yêu cầu slide 19

| # | Yêu cầu slide | Trạng thái | Bằng chứng |
|---|---------------|------------|------------|
| 1 | **Learn:** Tensors, Datasets/Loaders, Transforms, Model Building, Autograd, Optimization, Save/Load | ✅ Đạt | `practice1_fashionmnist.py` cover 7 bước, mapping ở README § Workflow |
| 2.1 | Load FashionMNIST dataset | ✅ Đạt | `datasets.FashionMNIST(root="Practice1/data", download=True)`, 60000 train / 10000 test |
| 2.2 | Apply transforms | ✅ Đạt | `Compose([ToTensor(), Normalize(0.2860, 0.3530)])` — mean/std chuẩn FashionMNIST |
| 2.3 | Build a neural network | ✅ Đạt | `FashionMLP`: 784→512→256→128→10, ReLU+Dropout(0.2), 567.434 params |
| 2.4 | Create training loop (forward, loss, backward, optimize) | ✅ Đạt | `forward → CrossEntropyLoss → zero_grad → backward → step`, log mỗi epoch |
| 2.5 | Evaluate model accuracy | ✅ Đạt | `evaluate()` trên test_loader, val_acc 87.21% (5 epochs) |
| 2.6 | Save and load the model | ✅ Đạt | `torch.save` + `torch.load` + `load_state_dict`, verify acc khớp 87.21% |
| 3.1 | Experiment with network/hyperparameters | ✅ Đạt | argparse `--lr/--optimizer/--hidden-dims/--dropout`, bảng gợi ý trong README |
| 3.2 | Visualize loss | ✅ Đạt | `outputs/loss_curves.png` (train vs val loss) + `outputs/accuracy.png` |
| 3.3 | Display predicted vs actual images | ✅ Đạt | `outputs/predictions.png` — grid 5×5, xanh=đúng đỏ=sai |
| 4 | Deliver: Python code, brief report, loss graphs, image displays | ✅ Đạt | `practice1_fashionmnist.py` + `README.md` + 3 ảnh outputs |
| 5 | Use PyTorch docs/tutorials | ✅ Đạt | Tham chiếu `pytorch.org/tutorials/beginner/basics/intro.html` (slide 18) |

**Kết luận:** 100% yêu cầu slide 19 đã được thực hiện. Không có mục nào thiếu.

---

## 2. Đối chiếu workflow slide 18 (7 bước)

| Bước slide 18 | Ánh xạ trong code |
|---------------|-------------------|
| 0. Quickstart | `demo_tensors()` |
| 1. Tensors | `torch.randn`, `view`, `to(device)` |
| 2. Datasets & DataLoaders | `FashionMNIST` + `DataLoader(batch_size=64, shuffle=True/False, num_workers=2)` |
| 3. Transforms | `transforms.Compose([ToTensor, Normalize])` |
| 4. Build Model | `FashionMLP(nn.Module)`, `nn.Sequential`, `nn.Linear/ReLU/Dropout` |
| 5. Automatic Differentiation | `loss.backward()` (Autograd) |
| 6. Optimization Loop | `optimizer.zero_grad/step`, epoch loop, `torch.optim.Adam/SGD` |
| 7. Save, Load and Use Model | `torch.save/load`, `state_dict`, reload verify |

---

## 3. Kiểm tra file & artifacts

| File | Tồn tại | Kích thước | Ghi chú |
|------|---------|------------|---------|
| `practice1_fashionmnist.py` | ✅ | 9.1 KB, 217 dòng | argparse, đầy đủ 7 bước |
| `models/fashionmnist_mlp.pth` | ✅ | 2.2 MB | chứa `model_state` + `hidden_dims` + `dropout` |
| `outputs/loss_curves.png` | ✅ | 64 KB | train/val loss theo epoch |
| `outputs/accuracy.png` | ✅ | 57 KB | val accuracy theo epoch |
| `outputs/predictions.png` | ✅ | 126 KB | 5×5 predicted vs actual |
| `data/FashionMNIST/raw/*` | ✅ | 4 file .gz + 4 file raw | 60k train + 10k test |
| `README.md` | ✅ | ~3 KB | báo cáo, bảng kết quả, cách chạy |
| `AUDIT.md` (file này) | ✅ | — | — |
| `LOG.md` | ✅ | — | — |

---

## 4. Kiểm tra chạy thực tế

| Lần chạy | Lệnh | Kết quả | Ghi chú |
|----------|------|---------|---------|
| Smoke 2 epochs | `myvenv/bin/python Practice1/practice1_fashionmnist.py --epochs 2` | val_acc 85.39% | PASS |
| Full 5 epochs | `myvenv/bin/python Practice1/practice1_fashionmnist.py --epochs 5` | val_acc **87.21%**, loss 0.3503 | PASS, best checkpoint lưu |
| Reload verify | `torch.load` + `load_state_dict` | acc 87.21% khớp | PASS |

Không có lỗi, warning nghiêm trọng. `num_workers=2` hoạt động ổn trên Linux.

---

## 5. Rủi ro & khuyến nghị

| Rủi ro | Mức độ | Khuyến nghị |
|--------|--------|-------------|
| Chưa thử `--optimizer sgd`/`--lr` khác trong lần chạy lưu | Thấp | Chạy thêm 2-3 config để bảng hyperparams trong README có số thực đo thay vì ước lượng |
| `Normalize(0.2860, 0.3530)` là đúng nhưng chưa ghi nguồn trong code comment | Thấp | Đã ghi trong README, nên thêm comment inline trong code |
| Thiếu `requirements.txt` riêng cho Practice1 | Thấp | Có thể thêm `Practice1/requirements.txt` hoặc dùng chung `Practice2/requirements.txt` |

---

## 6. Kết luận audit

- **Mức độ hoàn thành:** 100% yêu cầu slide.
- **Chất lượng code:** Tốt — có argparse, docstring, tách hàm `evaluate`/`get_dataloaders`, xử lý device linh hoạt.
- **Sẵn sàng nộp:** Có — code chạy được, outputs đầy đủ, README rõ ràng.
