"""
PRACTICE 2 — explore_model.py
===============================
Step 2 + 3 + 4: load từng pre-trained model, in kiến trúc ra file,
đếm tổng số tham số và số tham số của lớp phân loại cuối.

Chạy: myvenv/bin/python practice2/explore_model.py
"""

import io
import os
import sys

import torch
import torch.nn as nn
from torchvision import models

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "logs")
os.makedirs(OUT, exist_ok=True)

REGISTRY = {
    "resnet18":    (models.resnet18,    models.ResNet18_Weights.DEFAULT,
                    "fc"),
    "vgg16":       (models.vgg16,       models.VGG16_Weights.DEFAULT,
                    "classifier[6]"),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT,
                    "classifier"),
}


def head_params(model, head_attr):
    if head_attr == "fc":
        return model.fc
    if head_attr == "classifier":
        return model.classifier
    if head_attr == "classifier[6]":
        return model.classifier[6]


def main():
    print(f"{'model':<13} {'weights':<10} {'total params':>13} {'head':<16} {'head params':>12}")
    print("-" * 70)
    for name, (factory, weights_enum, head_attr) in REGISTRY.items():
        try:
            model = factory(weights=weights_enum)
            wtag = "pre-trained"
        except Exception as e:
            print(f"  !! {name}: không tải được weights ({type(e).__name__}) -> dùng weights=None")
            model = factory(weights=None)
            wtag = "random"

        buf = io.StringIO()
        print(model, file=buf)
        with open(os.path.join(OUT, f"arch_{name}.txt"), "w") as f:
            f.write(buf.getvalue())

        total = sum(p.numel() for p in model.parameters())
        head = head_params(model, head_attr)
        hparams = sum(p.numel() for p in head.parameters())

        # Demo Step 4: thay head cho 10 lớp CIFAR-10
        if head_attr == "fc":
            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, 10)
        elif head_attr == "classifier":
            in_f = model.classifier.in_features
            model.classifier = nn.Linear(in_f, 10)
        else:  # vgg classifier[6]
            in_f = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_f, 10)

        new_total = sum(p.numel() for p in model.parameters())
        print(f"{name:<13} {wtag:<10} {total:>13,} {head_attr:<16} {hparams:>12,}")
        print(f"{'':<13} {'':<10} {'-> sau thay head (Step 4):':<40} {new_total:>12,}")

        # Forward test
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 10), out.shape
        print(f"{'':<13} forward [1,3,224,224] -> {tuple(out.shape)} OK")
    print(f"\nKiến trúc đầy đủ đã lưu tại {OUT}/arch_*.txt (đem dán vào báo cáo)")


if __name__ == "__main__":
    main()
