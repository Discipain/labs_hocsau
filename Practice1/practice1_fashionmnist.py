"""
Practice 1 — PyTorch FashionMNIST Classification (Chapter 2 - ANN)
==================================================================
Workflow theo slide 18 + Tasks slide 19:
  0. Tensors, 1. Datasets/DataLoaders, 2. Transforms, 3. Build Model,
  4. Autograd, 5. Optimization Loop, 6. Evaluate, 7. Save/Load
  + Experiment hyperparams, Visualize loss, Display predicted vs actual.

Chạy:
  myvenv/bin/python Practice1/practice1_fashionmnist.py --epochs 5
  myvenv/bin/python Practice1/practice1_fashionmnist.py --epochs 2 --batch-size 64 --lr 1e-3
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Config ──────────────────────────────────────────────────────────
CLASS_NAMES = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0. Tensors — demo nhỏ
def demo_tensors(device):
    print("\n[0] Tensors demo")
    x = torch.randn(2, 3, device=device)
    print(f"  Tensor shape {x.shape}, device {x.device}")

# 3. Build Model — MLP 784 -> 512 -> 256 -> 128 -> 10
class FashionMLP(nn.Module):
    def __init__(self, hidden_dims=(512, 256, 128), dropout=0.2):
        super().__init__()
        layers = []
        in_dim = 28 * 28
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten 28x28 -> 784
        x = self.features(x)
        return self.classifier(x)  # logits, dùng CrossEntropyLoss

def get_dataloaders(data_root, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # mean/std FashionMNIST
    ])
    train_ds = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    test_ds  = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=2)
    return train_ds, test_ds, train_loader, test_loader, transform

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total

def train(args):
    device = get_device()
    print(f"Device: {device}")
    demo_tensors(device)

    data_root = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_root, exist_ok=True)
    train_ds, test_ds, train_loader, test_loader, transform = get_dataloaders(data_root, args.batch_size)
    print(f"\n[1-2] Dataset: train={len(train_ds)}, test={len(test_ds)}, batch_size={args.batch_size}")
    # show 1 batch
    imgs, lbls = next(iter(train_loader))
    print(f"  Batch shape: {imgs.shape}, labels {lbls[:8].tolist()}")

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))
    model = FashionMLP(hidden_dims=hidden_dims, dropout=args.dropout).to(device)
    print(f"\n[3] Model: hidden_dims={hidden_dims}, dropout={args.dropout}")
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    print(f"\n[4-5] Training: epochs={args.epochs}, optimizer={args.optimizer}, lr={args.lr}")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()       # Autograd
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_ds)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch:2d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%")

    print(f"\n[6] Best val_acc: {best_acc*100:.2f}%")

    # 7. Save / Load
    out_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "fashionmnist_mlp.pth")
    torch.save({"model_state": best_state, "hidden_dims": hidden_dims, "dropout": args.dropout}, ckpt)
    print(f"[7] Saved to {ckpt}")

    # Verify load
    loaded = torch.load(ckpt, map_location=device)
    model2 = FashionMLP(hidden_dims=tuple(loaded["hidden_dims"]), dropout=loaded["dropout"]).to(device)
    model2.load_state_dict(loaded["model_state"])
    _, acc2 = evaluate(model2, test_loader, criterion, device)
    print(f"  Reload verify acc: {acc2*100:.2f}%")

    # Visualize loss
    fig_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(fig_dir, exist_ok=True)
    epochs = range(1, args.epochs + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_loss"], marker="o", label="train_loss")
    plt.plot(epochs, history["val_loss"], marker="s", label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("FashionMNIST — Loss curves")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    p1 = os.path.join(fig_dir, "loss_curves.png")
    plt.savefig(p1, dpi=160); plt.close()
    print(f"  Saved {p1}")

    # Accuracy curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [a*100 for a in history["val_acc"]], marker="o", color="green")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("FashionMNIST — Validation Accuracy")
    plt.grid(alpha=0.3); plt.tight_layout()
    p_acc = os.path.join(fig_dir, "accuracy.png")
    plt.savefig(p_acc, dpi=160); plt.close()
    print(f"  Saved {p_acc}")

    # Predicted vs Actual — 5x5 grid
    # Lấy 25 ảnh test
    model.eval()
    # Lấy batch đầu test
    sample_loader = DataLoader(test_ds, batch_size=25, shuffle=True)
    images, labels = next(iter(sample_loader))
    with torch.no_grad():
        logits = model(images.to(device))
        preds = logits.argmax(1).cpu()

    # Denormalize để hiển thị
    def denorm(t):
        return (t * 0.3530 + 0.2860).clamp(0, 1)

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        img = denorm(images[i]).squeeze().numpy()
        plt.imshow(img, cmap="gray")
        color = "green" if preds[i].item() == labels[i].item() else "red"
        plt.title(f"P:{CLASS_NAMES[preds[i]][:6]}\nT:{CLASS_NAMES[labels[i]][:6]}", fontsize=7, color=color)
        plt.axis("off")
    plt.suptitle("Predicted (P) vs Actual (T) — green=correct, red=wrong", fontsize=11)
    plt.tight_layout()
    p2 = os.path.join(fig_dir, "predictions.png")
    plt.savefig(p2, dpi=160); plt.close()
    print(f"  Saved {p2}")

    # Simple experiment log — ghi thêm 1 dòng so sánh
    print("\n[Experiment note]")
    print(f"  Config: hidden={hidden_dims}, lr={args.lr}, optimizer={args.optimizer}, dropout={args.dropout}")
    print(f"  => val_acc={best_acc*100:.2f}% — đổi --lr / --optimizer / --hidden-dims để so sánh (xem README).")

    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--hidden-dims", type=str, default="512,256,128", help="vd: 512,256,128")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    args = parser.parse_args()
    train(args)
