"""
PRACTICE 2 — train.py
======================
Script huấn luyện pre-trained model trên CIFAR-10.
Phủ Steps 2,4,5,6,7 của đề + Exercise 2/3 (tham số hóa, TensorBoard).

Cách dùng:
    myvenv/bin/python practice2/train.py --model resnet18 --mode transfer \
        --epochs 3 --lr 1e-3 --batch-size 32 --run-name exp1

    --model    : resnet18 | vgg16 | densenet121     (Step 2 — Exercise 1)
    --mode     : transfer (freeze hết backbone)     (Step 4)
                 finetune (mở khóa vài block cuối, lr backbone = lr/10)
    --device   : auto | cuda | cpu                  (tự phát hiện)
    --limit-train: chỉ dùng N ảnh train (debug nhanh)
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")  # không cần cửa sổ, xuất file
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "..", "data")   # CIFAR-10 đã tải ở đây từ trước
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

# ──────────────────────────────────────────────
# Step 2 + 4: load pre-trained, thay head, freeze/unfreeze
# ──────────────────────────────────────────────
MODEL_REGISTRY = {
    # name: (hàm tạo với weights, tên attribute head, hàm unfreeze khi finetune)
    "resnet18": (
        lambda w: models.resnet18(weights=w),
        lambda m, c: setattr(m, "fc", nn.Linear(m.fc.in_features, c)),
        lambda m: [m.layer4],
    ),
    "vgg16": (
        lambda w: models.vgg16(weights=w),
        lambda m, c: m.classifier.__setitem__(6, nn.Linear(4096, c)),
        lambda m: [m.features[24:]],   # block conv cuối
    ),
    "densenet121": (
        lambda w: models.densenet121(weights=w),
        lambda m, c: setattr(m, "classifier", nn.Linear(m.classifier.in_features, c)),
        lambda m: [m.features.denseblock4],
    ),
}

WEIGHTS_REGISTRY = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "vgg16": models.VGG16_Weights.DEFAULT,
    "densenet121": models.DenseNet121_Weights.DEFAULT,
}


def build_model(name: str, num_classes: int, mode: str, log):
    """Step 2: load pre-trained; Step 4: thay head + freeze/unfreeze."""
    factory, set_head, finetune_blocks = MODEL_REGISTRY[name]
    try:
        model = factory(WEIGHTS_REGISTRY[name])   # weights ImageNet (tự cache)
        log(f"  [weights] pre-trained ImageNet: OK (đã cache hoặc vừa tải)")
    except Exception as e:
        log(f"  [weights] !! KHÔNG tải được pre-trained ({type(e).__name__}: {e})")
        log(f"  [weights] --> fallback weights=NGUỶEN (train từ đầu, kết quả sẽ thấp hơn nhiều!)")
        model = factory(None)

    # Freeze TẤT CẢ trước, rồi thay head (head mới tự requires_grad=True) — đúng thứ tự Lab 06
    for p in model.parameters():
        p.requires_grad = False
    set_head(model, num_classes)

    if mode == "finetune":
        for block in finetune_blocks(model):
            for p in block.parameters():
                p.requires_grad = True
        log(f"  [mode] finetune: đã mở khóa block cuối của backbone")
    else:
        log(f"  [mode] transfer: backbone đông cứng, chỉ head mới được học")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  [params] total={total:,} | trainable={trainable:,} "
        f"({100*trainable/total:.2f}%)")
    return model


def build_optimizer(model, name, lr, mode):
    """Finetune: param groups — backbone lr/10, head lr đầy đủ (bài học Lab 06)."""
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and (n.startswith("fc.") or n.startswith("classifier."))]
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not (n.startswith("fc.") or n.startswith("classifier."))]

    if mode == "finetune" and backbone_params:
        groups = [{"params": backbone_params, "lr": lr / 10},
                  {"params": head_params, "lr": lr}]
    else:
        groups = [{"params": head_params, "lr": lr}]

    if name == "adam":
        return optim.Adam(groups, weight_decay=1e-4)  # L2 nhẹ — Chapter 3 slide 22
    return optim.SGD(groups, lr=lr, momentum=0.9)


# ──────────────────────────────────────────────
# Step 5: data
# ──────────────────────────────────────────────
def build_loaders(batch_size, limit_train, log):
    imagenet_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize(224),                    # CIFAR 32x32 -> 224x224
        transforms.RandomHorizontalFlip(),         # data augmentation đơn giản
        transforms.ToTensor(),
        transforms.Normalize(**imagenet_norm),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(**imagenet_norm),
    ])
    train_set = datasets.CIFAR10(DATA_ROOT, train=True, download=False, transform=train_tf)
    test_set = datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=test_tf)
    if limit_train:
        train_set = Subset(train_set, range(limit_train))
        log(f"  [data] giới hạn train còn {limit_train} ảnh (chế độ debug)")
    nw = 0 if limit_train else 4
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False,
                             num_workers=nw)
    log(f"  [data] CIFAR-10: train={len(train_set)}, test={len(test_set)}, "
        f"batch={batch_size}, Resize(224)+Normalize(ImageNet)")
    return train_loader, test_loader


# ──────────────────────────────────────────────
# Step 6: train — đúng 6 bước Chapter 3
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, writer, global_step):
    model.train()
    running, correct, total = 0.0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)   # 1. chuyển device
        optimizer.zero_grad()                                # 2. xóa gradient cũ
        output = model(data)                                 # 3. forward
        loss = criterion(output, target)                     # 4. loss
        loss.backward()                                      # 5. backward
        optimizer.step()                                     # 6. cập nhật weights
        running += loss.item() * target.size(0)
        correct += output.argmax(1).eq(target).sum().item()
        total += target.size(0)
        global_step += 1
        if global_step % 100 == 0:
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
    return running / total, 100 * correct / total, global_step


# ──────────────────────────────────────────────
# Step 7: evaluate — eval() + no_grad()
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, confusion=None):
    model.eval()
    running, correct, total = 0.0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        running += criterion(output, target).item() * target.size(0)
        pred = output.argmax(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        if confusion is not None:
            for t, p in zip(target.tolist(), pred.tolist()):
                confusion[t][p] += 1
    return running / total, 100 * correct / total


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18", choices=list(MODEL_REGISTRY))
    ap.add_argument("--mode", default="transfer", choices=["transfer", "finetune"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--limit-train", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)

    log("=" * 62)
    log(f"PRACTICE 2 — {args.model} | mode={args.mode} | optimizer={args.optimizer}")
    log(f"  epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")
    log(f"  device = {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (CPU-only)"))
    log(f"  run    = {args.run_name}")
    log("=" * 62)

    # Step 2 + 4
    log("\n[Step 2+4] Model")
    model = build_model(args.model, num_classes=10, mode=args.mode, log=log).to(dev)

    # Step 5
    log("\n[Step 5] Dữ liệu")
    train_loader, test_loader = build_loaders(args.batch_size, args.limit_train, log)

    # Step 6
    log("\n[Step 6] Huấn luyện")
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.mode)
    writer = SummaryWriter(os.path.join(HERE, "runs", args.run_name))
    dummy = torch.randn(1, 3, 224, 224).to(dev)
    try:
        writer.add_graph(model, dummy)
    except Exception as e:
        log(f"  (add_graph bỏ qua: {type(e).__name__})")

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        tr_loss, tr_acc, step = train_one_epoch(
            model, train_loader, criterion, optimizer, dev, writer, step)
        va_loss, va_acc = evaluate(model, test_loader, criterion, dev)
        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        hist["train_acc"].append(tr_acc);   hist["val_acc"].append(va_acc)
        writer.add_scalars("Epoch", {"train_loss": tr_loss, "val_loss": va_loss,
                                     "train_acc": tr_acc, "val_acc": va_acc}, epoch + 1)
        log(f"  Epoch {epoch+1}/{args.epochs} ({time.time()-t0:.0f}s): "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.1f}% | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.1f}%")
    writer.close()

    # Step 7 — đánh giá chi tiết + confusion matrix
    log("\n[Step 7] Đánh giá chi tiết")
    conf = [[0] * 10 for _ in range(10)]
    _, va_acc = evaluate(model, test_loader, criterion, dev, confusion=conf)
    log(f"  Val accuracy cuối: {va_acc:.2f}%")
    conf_np = np.array(conf)
    per_class = np.diag(conf_np) / conf_np.sum(axis=1).clip(min=1) * 100
    for i, c in enumerate(CLASSES):
        log(f"    {c:10s}: {per_class[i]:5.1f}%  (n={conf_np[i].sum()})")

    # Lưu artifacts
    ckpt = os.path.join(HERE, "checkpoints", f"{args.run_name}.pth")
    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "val_acc": va_acc}, ckpt)
    log(f"\n  [saved] checkpoint: {ckpt}")

    # 2 ảnh PNG cho báo cáo
    png = os.path.join(HERE, "logs", f"{args.run_name}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, args.epochs + 1)
    axes[0].plot(ep, hist["train_loss"], "o-", label="train")
    axes[0].plot(ep, hist["val_loss"], "s-", label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(ep, hist["train_acc"], "o-", label="train")
    axes[1].plot(ep, hist["val_acc"], "s-", label="val")
    axes[1].set_title("Accuracy (%)"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.suptitle(args.run_name)
    fig.tight_layout()
    fig.savefig(png + "_curves.png", dpi=110)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(conf_np, cmap="Blues")
    ax2.set_xticks(range(10), CLASSES, rotation=90, fontsize=7)
    ax2.set_yticks(range(10), CLASSES, fontsize=7)
    thresh = conf_np.max() / 2
    for i in range(10):
        for j in range(10):
            ax2.text(j, i, str(conf_np[i, j]), ha="center", va="center", fontsize=6,
                     color="white" if conf_np[i, j] > thresh else "black")
    ax2.set_title(f"Confusion matrix — {args.run_name} (val)")
    fig2.colorbar(im, fraction=0.046)
    fig2.tight_layout()
    fig2.savefig(png + "_cm.png", dpi=110)
    log(f"  [saved] {png}_curves.png, {png}_cm.png")

    with open(os.path.join(HERE, "logs", f"{args.run_name}.log"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log("\nHOÀN THÀNH ✓")


if __name__ == "__main__":
    main()
