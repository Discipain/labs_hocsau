"""
PRACTICE 2 — train.py
======================
Script huấn luyện pre-trained model trên CIFAR-10.
Phủ Steps 2,4,5,6,7 của đề + Exercise 2/3 (tham số hóa, TensorBoard).

Cách dùng:
    myvenv/bin/python practice2/train.py --model resnet18 --mode transfer \
        --epochs 3 --lr 1e-3 --batch-size 32 --run-name exp1

    --model    : resnet18 | vgg16 | densenet121     (Step 2 — Exercise 1)
                 Chọn kiến trúc CNN sẽ mượn từ ImageNet
    --mode     : transfer (freeze hết backbone)     (Step 4)
                 finetune (mở khóa vài block cuối, lr backbone = lr/10)
                 transfer = chỉ học head mới | finetune = học thêm block cuối
    --device   : auto | cuda | cpu                  (tự phát hiện GPU)
    --limit-train: chỉ dùng N ảnh train (debug nhanh, ví dụ 1000 ảnh)
"""

import argparse  # Thư viện phân tích tham số dòng lệnh (--model, --lr, ...)
import os        # Thao tác thư mục, đường dẫn file
import time      # Đo thời gian chạy mỗi epoch

import matplotlib
matplotlib.use("Agg")  # Chế độ vẽ không cần màn hình (vẽ thẳng ra file .png)
import matplotlib.pyplot as plt  # Vẽ đồ thị loss/accuracy
import numpy as np               # Tính toán ma trận (confusion matrix)
import torch                     # Thư viện Deep Learning chính
import torch.nn as nn            # Chứa các lớp Linear, Conv2d, CrossEntropyLoss, ...
import torch.optim as optim       # Chứa các bộ tối ưu Adam, SGD, ...
from torch.utils.data import DataLoader, Subset  # DataLoader: xe chở từng Batch | Subset: cắt nhỏ dataset để debug
from torch.utils.tensorboard import SummaryWriter # Ghi log cho TensorBoard (xem trên trình duyệt)
from torchvision import datasets, models, transforms # datasets: CIFAR-10 | models: ResNet/VGG/DenseNet | transforms: Resize, Normalize, ...

# ── Hằng số đường dẫn và nhãn ──────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))  # Thư mục chứa file train.py này (practice2/)
DATA_ROOT = os.path.join(HERE, "..", "data")       # Thư mục chứa CIFAR-10 (đã tải sẵn, không cần download lại)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
# CLASSES: 10 tên lớp của CIFAR-10, thứ tự 0-9 phải khớp với nhãn trong dataset
#          Dùng để đặt tên hàng/cột cho confusion matrix

# ──────────────────────────────────────────────
# Step 2 + 4: load pre-trained, thay head, freeze/unfreeze
# Đây là phần quan trọng nhất của Transfer Learning
# ──────────────────────────────────────────────

# MODEL_REGISTRY: Sổ đăng ký 3 kiến trúc — tránh viết if/elif dài dòng
# Mỗi model gồm 3 hàm lambda:
#   1. Hàm tạo model với weights (trọng số ImageNet)
#   2. Hàm thay head (lớp cuối) từ 1000 lớp -> 10 lớp
#   3. Hàm chỉ ra khối cuối sẽ được mở khóa khi --mode finetune
MODEL_REGISTRY = {
    "resnet18": (
        lambda w: models.resnet18(weights=w),  # Tạo ResNet18, w là bộ trọng số ImageNet (hoặc None nếu train từ đầu)
        # Thay head: model.fc là lớp cuối của ResNet (Fully Connected)
        # m.fc.in_features = 512 (số đầu vào của head cũ), c = 10 (số lớp mới)
        # nn.Linear(512, 10) = tạo head mới với 512*10 + 10 = 5130 tham số
        lambda m, c: setattr(m, "fc", nn.Linear(m.fc.in_features, c)),
        # Khi finetune, mở khóa layer4 (khối residual cuối cùng, chứa ~8 triệu params)
        lambda m: [m.layer4],
    ),
    "vgg16": (
        lambda w: models.vgg16(weights=w),
        # VGG16: head nằm ở classifier[6] (lớp thứ 6 trong dãy classifier, 2 lớp trước là 4096 nơ-ron)
        lambda m, c: m.classifier.__setitem__(6, nn.Linear(4096, c)),
        lambda m: [m.features[24:]],   # block conv cuối (từ layer 24 trở đi)
    ),
    "densenet121": (
        lambda w: models.densenet121(weights=w),
        # DenseNet121: head là classifier (1 lớp duy nhất)
        lambda m, c: setattr(m, "classifier", nn.Linear(m.classifier.in_features, c)),
        lambda m: [m.features.denseblock4],  # khối dense cuối cùng
    ),
}

# WEIGHTS_REGISTRY: Bộ trọng số ImageNet mặc định cho từng kiến trúc
# DEFAULT = bộ trọng số tốt nhất do PyTorch cung cấp (train trên 14 triệu ảnh ImageNet, 1000 lớp)
WEIGHTS_REGISTRY = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "vgg16": models.VGG16_Weights.DEFAULT,
    "densenet121": models.DenseNet121_Weights.DEFAULT,
}


def build_model(name: str, num_classes: int, mode: str, log):
    """Step 2: load pre-trained; Step 4: thay head + freeze/unfreeze.

    Args:
        name: tên kiến trúc (resnet18/vgg16/densenet121)
        num_classes: số lớp mới (10 cho CIFAR-10)
        mode: "transfer" (đông cứng backbone) hay "finetune" (mở block cuối)
        log: hàm ghi log ra màn hình + file
    """
    factory, set_head, finetune_blocks = MODEL_REGISTRY[name]  # Lấy 3 hàm tương ứng với kiến trúc đã chọn

    # ── Thử tải trọng số pre-trained ImageNet ──
    try:
        model = factory(WEIGHTS_REGISTRY[name])   # Tải trọng số ImageNet (tự cache, lần sau không cần tải lại)
        log(f"  [weights] pre-trained ImageNet: OK (đã cache hoặc vừa tải)")
    except Exception as e:
        # Nếu không có mạng hoặc lỗi tải -> fallback: tạo model rỗng (weights=None), train từ đầu
        # Kết quả sẽ rất thấp (~36% thay vì ~79%) — đã ghi trong report
        log(f"  [weights] !! KHÔNG tải được pre-trained ({type(e).__name__}: {e})")
        log(f"  [weights] --> fallback weights=NGUỶEN (train từ đầu, kết quả sẽ thấp hơn nhiều!)")
        model = factory(None)  # None = khởi tạo ngẫu nhiên, không có kiến thức ImageNet

    # ── Freeze (đông cứng) TẤT CẢ tham số trước, rồi thay head ──
    # requires_grad = False nghĩa là "không cần tính gradient, không cập nhật w này nữa"
    # Đây là cốt lõi của Transfer Learning: giữ nguyên kiến thức ImageNet
    for p in model.parameters():
        p.requires_grad = False  # Đông cứng toàn bộ backbone (hàng triệu params)

    # Thay head mới: head mới được tạo bằng nn.Linear nên mặc định requires_grad=True
    # THỨ TỰ QUAN TRỌNG: phải freeze trước rồi mới thay head, nếu không head mới cũng bị freeze luôn!
    # (Bài học từ Lab 06)
    set_head(model, num_classes)

    # ── Nếu finetune: mở khóa thêm block cuối của backbone ──
    if mode == "finetune":
        for block in finetune_blocks(model):  # Lấy block cuối (ví dụ layer4 của ResNet18)
            for p in block.parameters():
                p.requires_grad = True  # Cho phép block này được học tiếp
        log(f"  [mode] finetune: đã mở khóa block cuối của backbone")
    else:
        log(f"  [mode] transfer: backbone đông cứng, chỉ head mới được học")

    # ── Đếm tham số để báo cáo ──
    total = sum(p.numel() for p in model.parameters())  # Tổng số w,b trong toàn model
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Số w,b được học (requires_grad=True)
    # Ví dụ ResNet18 transfer: total=11,181,642 | trainable=5,130 (0.05%)
    #        ResNet18 finetune: total=11,181,642 | trainable=8,398,858 (75.11%)
    log(f"  [params] total={total:,} | trainable={trainable:,} "
        f"({100*trainable/total:.2f}%)")
    return model


def build_optimizer(model, name, lr, mode):
    """Tạo bộ tối ưu (optimizer) với param groups.

    Ý tưởng: head mới cần học nhanh (lr=1e-3), backbone đã pre-trained chỉ cần chỉnh nhẹ (lr/10 = 1e-4)
    để tránh phá hủy kiến thức ImageNet (catastrophic forgetting).
    """
    # Tách riêng 2 nhóm tham số dựa trên tên:
    # head_params: các w có tên bắt đầu bằng "fc." hoặc "classifier." (lớp cuối mới thay)
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and (n.startswith("fc.") or n.startswith("classifier."))]
    # backbone_params: các w còn lại nhưng vẫn requires_grad=True (chỉ có khi finetune)
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not (n.startswith("fc.") or n.startswith("classifier."))]

    # Nếu finetune và có backbone được mở khóa -> tạo 2 nhóm với lr khác nhau
    if mode == "finetune" and backbone_params:
        groups = [{"params": backbone_params, "lr": lr / 10},  # Backbone học chậm hơn 10 lần
                  {"params": head_params, "lr": lr}]           # Head học với lr đầy đủ
    else:
        # Transfer: chỉ có head được học
        groups = [{"params": head_params, "lr": lr}]

    # Chọn optimizer: Adam (thông minh, tự điều chỉnh lr từng w) hay SGD (cổ điển, có momentum)
    if name == "adam":
        return optim.Adam(groups, weight_decay=1e-4)  # weight_decay=1e-4: L2 regularization, phạt w quá to để chống overfitting (Chapter 3 slide 22)
    return optim.SGD(groups, lr=lr, momentum=0.9)  # momentum=0.9: quán tính 90%, giúp vượt qua chỗ gồ ghề


# ──────────────────────────────────────────────
# Step 5: Chuẩn bị dữ liệu (data preparation)
# ──────────────────────────────────────────────
def build_loaders(batch_size, limit_train, log):
    """Tạo DataLoader cho train và test.

    Hai điểm BẮT BUỘC khi dùng pre-trained:
      1. Resize về đúng kích thước lúc pre-train (224x224 cho ImageNet)
      2. Normalize bằng đúng mean/std ImageNet (không phải FashionMNIST)
    Sai một trong hai là trọng số pre-trained mất ý nghĩa!
    """
    # Chuẩn ImageNet: mean và std tính trên 14 triệu ảnh ImageNet (3 kênh RGB)
    # Khác với FashionMNIST chỉ có 1 kênh xám với mean=0.2860
    imagenet_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Transform cho tập train: có thêm augmentation (tăng cường dữ liệu)
    train_tf = transforms.Compose([
        transforms.Resize(224),                    # CIFAR gốc 32x32 -> phóng to 224x224 (bắt buộc vì pre-trained yêu cầu 224)
                                                  # Nhược điểm: ảnh bị mờ, nhưng đề bài yêu cầu dùng pre-trained nên phải chấp nhận
        transforms.RandomHorizontalFlip(),         # Lật ảnh trái-phải ngẫu nhiên 50% — giúp model thấy nhiều biến thể, chống học vẹt
                                                  # Chỉ dùng cho train, test không lật để kết quả ổn định
        transforms.ToTensor(),                     # Đổi ảnh 0-255 (PIL) -> 0-1 (Tensor), đồng thời đổi shape [H,W,C] -> [C,H,W]
        transforms.Normalize(**imagenet_norm),     # Chuẩn hóa (x - mean)/std để dữ liệu có trung bình ~0, giúp hội tụ nhanh
    ])
    # Transform cho tập test: KHÔNG có RandomHorizontalFlip (giữ nguyên ảnh để đánh giá ổn định)
    test_tf = transforms.Compose([
        transforms.Resize(224),                    # Vẫn phải resize 224 để khớp với train
        transforms.ToTensor(),
        transforms.Normalize(**imagenet_norm),     # Vẫn phải normalize cùng mean/std ImageNet
    ])

    # Tải CIFAR-10: train=True -> 50.000 ảnh train | train=False -> 10.000 ảnh test
    # download=False vì data đã có sẵn ở DATA_ROOT, không cần tải lại
    train_set = datasets.CIFAR10(DATA_ROOT, train=True, download=False, transform=train_tf)
    test_set = datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=test_tf)

    # Chế độ debug: chỉ lấy N ảnh đầu để chạy thử nhanh (ví dụ --limit-train 1000)
    if limit_train:
        train_set = Subset(train_set, range(limit_train))  # Cắt nhỏ dataset, chỉ lấy limit_train ảnh đầu
        log(f"  [data] giới hạn train còn {limit_train} ảnh (chế độ debug)")

    # num_workers: số tiến trình phụ chuẩn bị data song song
    # 4 khi chạy thật (nhanh), 0 khi debug (tránh lỗi đa tiến trình)
    nw = 0 if limit_train else 4

    # DataLoader train: shuffle=True (xáo trộn mỗi epoch để model không học vẹt thứ tự)
    # batch_size=32 nghĩa là mỗi lần chở 32 ảnh lên GPU
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=False, drop_last=False)
    # DataLoader test: batch_size*2 (gấp đôi vì khi test không cần backward nên tốn ít bộ nhớ hơn, chở nhiều cho nhanh)
    # shuffle=False (giữ nguyên thứ tự để confusion matrix đúng)
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False,
                             num_workers=nw)
    log(f"  [data] CIFAR-10: train={len(train_set)}, test={len(test_set)}, "
        f"batch={batch_size}, Resize(224)+Normalize(ImageNet)")
    return train_loader, test_loader


# ──────────────────────────────────────────────
# Step 6: Huấn luyện — đúng 6 bước chuẩn Chapter 3
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, writer, global_step):
    """Huấn luyện 1 epoch (1 lần duyệt hết 50.000 ảnh train).

    6 bước chuẩn:
      1. Chuyển data lên device (GPU/CPU)
      2. Xóa gradient cũ (zero_grad)
      3. Forward (cho ảnh qua model để đoán)
      4. Tính loss (so sánh đoán với nhãn thật)
      5. Backward (lan truyền ngược tìm lỗi)
      6. Cập nhật weights (optimizer.step)

    Returns:
        train_loss, train_acc, global_step mới
    """
    model.train()  # Báo cho model: đang ở chế độ huấn luyện -> bật Dropout/BatchNorm ở chế độ train
    running, correct, total = 0.0, 0, 0  # running: tổng loss | correct: số đoán đúng | total: tổng số ảnh đã duyệt
    for data, target in loader:  # Lặp qua từng Batch (ví dụ 50.000/32 = 1562 Batch)
        data, target = data.to(device), target.to(device)   # 1. Chuyển Batch ảnh [32,3,224,224] và nhãn [32] lên GPU/CPU
        optimizer.zero_grad()                                # 2. Xóa gradient cũ (PyTorch có thói quen cộng dồn gradient, không xóa sẽ sai!)
        output = model(data)                                 # 3. Forward: cho 32 ảnh qua ResNet/VGG/DenseNet -> ra 32x10 điểm logits
        loss = criterion(output, target)                     # 4. Tính loss: CrossEntropyLoss so sánh 10 điểm logits với nhãn thật 0-9
        loss.backward()                                      # 5. Backward: Autograd tự tính gradient cho từng w (lan truyền ngược từ loss về lớp đầu)
        optimizer.step()                                     # 6. Cập nhật: Adam/SGD sửa từng w một chút theo gradient và lr (head: lr=1e-3, backbone finetune: lr=1e-4)
        running += loss.item() * target.size(0)  # Cộng dồn loss (nhân với batch size vì loss là trung bình của batch)
        correct += output.argmax(1).eq(target).sum().item()  # Đếm số đoán đúng: argmax lấy lớp có điểm cao nhất, so với nhãn thật
        total += target.size(0)  # Cộng dồn tổng số ảnh
        global_step += 1  # Đếm tổng số Batch đã chạy (dùng cho TensorBoard)
        if global_step % 100 == 0:
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)  # Mỗi 100 Batch ghi loss lên TensorBoard để vẽ đường cong mịn
    return running / total, 100 * correct / total, global_step  # Trả về loss trung bình và accuracy % của cả epoch


# ──────────────────────────────────────────────
# Step 7: Đánh giá — eval() + no_grad()
# ──────────────────────────────────────────────
@torch.no_grad()  # Decorator: tắt Autograd hoàn toàn trong hàm này -> tiết kiệm 50% bộ nhớ, chạy nhanh hơn (vì không cần backward)
def evaluate(model, loader, criterion, device, confusion=None):
    """Đánh giá model trên tập test (10.000 ảnh).

    Args:
        confusion: ma trận 10x10, confusion[t][p] = số ảnh nhãn thật t nhưng đoán thành p
                   Nếu None thì không ghi confusion matrix

    Returns:
        val_loss, val_acc (%)
    """
    model.eval()  # Báo cho model: đang ở chế độ đánh giá -> tắt Dropout, BatchNorm dùng trung bình đã học (không cập nhật nữa)
    running, correct, total = 0.0, 0, 0
    for data, target in loader:  # Lặp qua từng Batch test (không xáo trộn)
        data, target = data.to(device), target.to(device)  # Chuyển lên device
        output = model(data)  # Forward (chỉ forward, không backward vì có @no_grad)
        running += criterion(output, target).item() * target.size(0)  # Cộng dồn loss
        pred = output.argmax(1)  # Lấy nhãn đoán (lớp có điểm cao nhất trong 10)
        correct += pred.eq(target).sum().item()  # Đếm số đoán đúng
        total += target.size(0)
        if confusion is not None:
            for t, p in zip(target.tolist(), pred.tolist()):  # Duyệt từng cặp (nhãn thật, nhãn đoán)
                confusion[t][p] += 1  # Tăng ô [t][p] trong ma trận 10x10
                # Ví dụ: ảnh mèo (t=3) đoán thành chó (p=5) -> confusion[3][5] += 1
                # Đường chéo confusion[i][i] là số đoán đúng của lớp i
    return running / total, 100 * correct / total  # Loss trung bình và accuracy % trên toàn bộ test


# ──────────────────────────────────────────────
# Main: Điều phối toàn bộ pipeline
# ──────────────────────────────────────────────
def main():
    # ── Định nghĩa tham số dòng lệnh ──
    ap = argparse.ArgumentParser()  # Bộ phân tích tham số dòng lệnh
    ap.add_argument("--model", default="resnet18", choices=list(MODEL_REGISTRY))  # Chọn kiến trúc
    ap.add_argument("--mode", default="transfer", choices=["transfer", "finetune"])  # Chọn chiến lược
    ap.add_argument("--epochs", type=int, default=3)       # Số epoch (số lần duyệt hết 50k ảnh)
    ap.add_argument("--lr", type=float, default=1e-3)      # Learning rate (độ dài sải chân)
    ap.add_argument("--batch-size", type=int, default=32)  # Số ảnh mỗi Batch
    ap.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])  # Bộ tối ưu
    ap.add_argument("--run-name", required=True)  # Tên run (bắt buộc) — dùng để đặt tên file log, checkpoint, ảnh PNG
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])  # Chọn GPU hay CPU
    ap.add_argument("--limit-train", type=int, default=0)  # 0 = dùng hết 50k ảnh, >0 = chỉ dùng N ảnh đầu (debug)
    args = ap.parse_args()  # Phân tích tham số người dùng gõ

    # ── Tạo thư mục lưu kết quả ──
    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)  # Chứa file .pth (trọng số đã học)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)         # Chứa ảnh PNG và file .log
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)         # Chứa event TensorBoard

    # ── Hệ thống log: vừa in ra màn hình vừa lưu vào file ──
    log_lines = []  # Danh sách các dòng log để cuối cùng ghi ra file

    def log(msg):
        print(msg)            # In ra terminal
        log_lines.append(msg) # Lưu vào danh sách

    # ── Chọn device (GPU hay CPU) ──
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"  # Tự phát hiện: có GPU thì dùng GPU, không thì CPU
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"  # Fallback nếu người dùng ép --device cuda nhưng máy không có GPU
    dev = torch.device(device)  # Đối tượng device để dùng cho .to(dev)

    # ── In header ──
    log("=" * 62)
    log(f"PRACTICE 2 — {args.model} | mode={args.mode} | optimizer={args.optimizer}")
    log(f"  epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")
    log(f"  device = {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (CPU-only)"))
    log(f"  run    = {args.run_name}")  # Tên run, ví dụ ex1_resnet18_transfer
    log("=" * 62)

    # ── Step 2 + 4: Tạo model ──
    log("\n[Step 2+4] Model")
    model = build_model(args.model, num_classes=10, mode=args.mode, log=log).to(dev)
    # .to(dev) = chuyển toàn bộ 11 triệu tham số lên GPU/CPU đã chọn

    # ── Step 5: Tạo DataLoader ──
    log("\n[Step 5] Dữ liệu")
    train_loader, test_loader = build_loaders(args.batch_size, args.limit_train, log)

    # ── Step 6: Chuẩn bị huấn luyện ──
    log("\n[Step 6] Huấn luyện")
    criterion = nn.CrossEntropyLoss()  # Hàm mất mát cho bài phân loại 10 lớp (đã bao gồm Softmax bên trong)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.mode)  # Bộ tối ưu với param groups
    writer = SummaryWriter(os.path.join(HERE, "runs", args.run_name))  # Ghi log cho TensorBoard
    dummy = torch.randn(1, 3, 224, 224).to(dev)  # Ảnh giả 1x3x224x224 để vẽ kiến trúc model
    try:
        writer.add_graph(model, dummy)  # Vẽ sơ đồ khối ResNet/VGG/DenseNet trong tab Graphs của TensorBoard
    except Exception as e:
        log(f"  (add_graph bỏ qua: {type(e).__name__})")  # Một số model (VGG) có thể lỗi khi vẽ graph, bỏ qua không sao

    # ── Vòng lặp huấn luyện qua từng epoch ──
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}  # Lịch sử để vẽ đồ thị
    step = 0  # Đếm tổng số Batch đã chạy (global_step cho TensorBoard)
    t0 = time.time()  # Thời điểm bắt đầu để tính thời gian
    for epoch in range(args.epochs):  # Lặp qua từng epoch (0, 1, 2, ... epochs-1)
        tr_loss, tr_acc, step = train_one_epoch(
            model, train_loader, criterion, optimizer, dev, writer, step)  # Huấn luyện 1 epoch -> trả về loss/acc train
        va_loss, va_acc = evaluate(model, test_loader, criterion, dev)  # Đánh giá trên test -> trả về loss/acc val
        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)  # Lưu vào lịch sử
        hist["train_acc"].append(tr_acc);   hist["val_acc"].append(va_acc)
        writer.add_scalars("Epoch", {"train_loss": tr_loss, "val_loss": va_loss,
                                     "train_acc": tr_acc, "val_acc": va_acc}, epoch + 1)  # Ghi lên TensorBoard
        log(f"  Epoch {epoch+1}/{args.epochs} ({time.time()-t0:.0f}s): "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.1f}% | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.1f}%")  # In ra màn hình
    writer.close()  # Đóng TensorBoard writer

    # ── Step 7: Đánh giá chi tiết + confusion matrix ──
    log("\n[Step 7] Đánh giá chi tiết")
    conf = [[0] * 10 for _ in range(10)]  # Ma trận 10x10 khởi tạo toàn 0
    _, va_acc = evaluate(model, test_loader, criterion, dev, confusion=conf)  # Đánh giá lần cuối và điền confusion matrix
    log(f"  Val accuracy cuối: {va_acc:.2f}%")
    conf_np = np.array(conf)  # Đổi sang numpy array để tính toán
    # Tính accuracy từng lớp: đường chéo / tổng hàng * 100
    # np.diag(conf_np) = [số đoán đúng lớp 0, lớp 1, ..., lớp 9] (đường chéo)
    # conf_np.sum(axis=1) = [tổng số ảnh lớp 0, lớp 1, ..., lớp 9] (tổng mỗi hàng)
    per_class = np.diag(conf_np) / conf_np.sum(axis=1).clip(min=1) * 100
    for i, c in enumerate(CLASSES):
        log(f"    {c:10s}: {per_class[i]:5.1f}%  (n={conf_np[i].sum()})")  # In accuracy từng lớp

    # ── Lưu checkpoint (trọng số đã học) ──
    ckpt = os.path.join(HERE, "checkpoints", f"{args.run_name}.pth")
    torch.save({"state_dict": model.state_dict(),  # state_dict: cuốn sổ chứa toàn bộ w,b đã học
                "args": vars(args),   # Lưu luôn tham số dòng lệnh để sau này biết run này chạy với lr nào
                "val_acc": va_acc},   # Lưu accuracy cuối để so sánh
               ckpt)
    log(f"\n  [saved] checkpoint: {ckpt}")  # Ví dụ: checkpoints/ex1_resnet18_transfer.pth

    # ── Vẽ 2 ảnh PNG cho báo cáo ──
    # Ảnh 1: Đường cong Loss và Accuracy theo epoch
    png = os.path.join(HERE, "logs", f"{args.run_name}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))  # 1 hàng 2 cột
    ep = range(1, args.epochs + 1)  # [1, 2, 3, ..., epochs]
    axes[0].plot(ep, hist["train_loss"], "o-", label="train")  # Đường train_loss: dấu tròn + nối liền
    axes[0].plot(ep, hist["val_loss"], "s-", label="val")      # Đường val_loss: dấu vuông + nối liền
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()  # Tiêu đề, nhãn trục, chú giải
    axes[1].plot(ep, hist["train_acc"], "o-", label="train")   # Tương tự cho Accuracy
    axes[1].plot(ep, hist["val_acc"], "s-", label="val")
    axes[1].set_title("Accuracy (%)"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.suptitle(args.run_name)  # Tiêu đề chung là tên run
    fig.tight_layout()  # Tự động căn chỉnh khoảng cách
    fig.savefig(png + "_curves.png", dpi=110)  # Lưu: logs/ex1_resnet18_transfer_curves.png

    # Ảnh 2: Confusion matrix (ma trận nhầm lẫn) 10x10
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(conf_np, cmap="Blues")  # Vẽ ma trận màu xanh: ô càng đậm = càng nhiều ảnh
    ax2.set_xticks(range(10), CLASSES, rotation=90, fontsize=7)  # Nhãn cột (nhãn đoán): xoay 90 độ
    ax2.set_yticks(range(10), CLASSES, fontsize=7)               # Nhãn hàng (nhãn thật)
    thresh = conf_np.max() / 2  # Ngưỡng để đổi màu chữ (ô đậm thì chữ trắng, ô nhạt thì chữ đen cho dễ đọc)
    for i in range(10):
        for j in range(10):
            ax2.text(j, i, str(conf_np[i, j]), ha="center", va="center", fontsize=6,
                     color="white" if conf_np[i, j] > thresh else "black")  # Ghi số vào từng ô
    ax2.set_title(f"Confusion matrix — {args.run_name} (val)")  # Tiêu đề
    fig2.colorbar(im, fraction=0.046)  # Thanh màu bên phải
    fig2.tight_layout()
    fig2.savefig(png + "_cm.png", dpi=110)  # Lưu: logs/ex1_resnet18_transfer_cm.png
    log(f"  [saved] {png}_curves.png, {png}_cm.png")

    # ── Lưu file log text ──
    with open(os.path.join(HERE, "logs", f"{args.run_name}.log"), "w") as f:
        f.write("\n".join(log_lines) + "\n")  # Ghi toàn bộ log ra file text để nộp kèm báo cáo
    log("\nHOÀN THÀNH ✓")  # Báo xong


if __name__ == "__main__":
    main()  # Chỉ chạy main() khi file được gọi trực tiếp (không chạy khi bị import)
