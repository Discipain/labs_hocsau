# Practice 1 — FashionMNIST Classification với PyTorch (Chapter 2 - ANN)

## Yêu cầu (slide 19)
- Học workflow PyTorch: Tensors, Datasets/DataLoaders, Transforms, Build Model, Autograd, Optimization Loop, Save/Load
- Implement: load FashionMNIST, transforms, MLP, training loop, evaluate, save/load
- Tasks: thí nghiệm hyperparams, visualize loss, hiển thị predicted vs actual
- Deliver: Python code, báo cáo ngắn, loss graphs, image displays

## Cách chạy
```bash
/home/toi/Documents/hocsau/Practice2/myvenv/bin/python Practice1/practice1_fashionmnist.py --epochs 5
# Tuỳ chọn: --batch-size 64 --lr 1e-3 --optimizer adam --hidden-dims 512,256,128 --dropout 0.2
```

## Kiến trúc Model
- **MLP**: 784 (28×28 flatten) → 512 → 256 → 128 (ReLU + Dropout 0.2) → 10 logits
- **Params**: 567.434
- **Loss**: CrossEntropyLoss | **Optimizer**: Adam (lr=1e-3, mặc định) / SGD (tuỳ chọn)
- **Transforms**: `ToTensor()` + `Normalize(mean=0.2860, std=0.3530)` (thống kê FashionMNIST)

## Kết quả (chạy thực tế, CUDA, 5 epochs)
| Epoch | train_loss | val_loss | val_acc |
|-------|-----------|----------|---------|
| 1 | 0.5195 | 0.4047 | 85.51% |
| 2 | 0.3907 | 0.3784 | 86.16% |
| 3 | 0.3568 | 0.3654 | 86.64% |
| 4 | 0.3341 | 0.3540 | 87.15% |
| 5 | 0.3180 | 0.3503 | **87.21%** |

- Best val_acc **87.21%**, checkpoint: `models/fashionmnist_mlp.pth` (có verify reload).

## Thí nghiệm Hyperparameter (gợi ý)
| Cấu hình | val_acc (2 epochs) | Ghi chú |
|----------|-------------------|---------|
| Adam lr=1e-3, hidden 512/256/128 | ~85.4% | baseline |
| Adam lr=5e-4 | ~84% (hội tụ chậm hơn) | giảm lr cần nhiều epoch hơn |
| SGD lr=1e-2 momentum 0.9 | ~82-84% | SGD hội tụ chậm hơn Adam |
| hidden 256,128 (nhỏ hơn) | ~84-85% | ít params, acc thấp hơn nhẹ |

Chạy so sánh:
```bash
python Practice1/practice1_fashionmnist.py --epochs 5 --lr 5e-4
python Practice1/practice1_fashionmnist.py --epochs 5 --optimizer sgd --lr 0.01
python Practice1/practice1_fashionmnist.py --epochs 5 --hidden-dims 256,128
```

## Outputs
- `outputs/loss_curves.png` — train/val loss theo epoch
- `outputs/accuracy.png` — val accuracy theo epoch
- `outputs/predictions.png` — grid 5×5 predicted (P) vs actual (T), xanh=đúng, đỏ=sai
- `models/fashionmnist_mlp.pth` — state_dict + config (load lại để inference)

## Workflow mapping (slide 18)
1. Tensors — `demo_tensors()` 
2. Datasets/DataLoaders — `FashionMNIST` + `DataLoader`
3. Transforms — `Compose([ToTensor, Normalize])`
4. Build Model — `FashionMLP(nn.Module)`
5. Autograd — `loss.backward()`
6. Optimization Loop — `optimizer.zero_grad/step`, loop qua epochs
7. Save/Load — `torch.save/load` + `torch.nn.Module.load_state_dict`
