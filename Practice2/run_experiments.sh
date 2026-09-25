#!/usr/bin/env bash
# PRACTICE 2 — chạy tuần tự 7 experiments (CPU, kế hoạch kịch bản B)
# Mỗi experiment độc lập: hỏng cái nào chạy lại riêng cái đó.
# Bật:  cd /home/toi/Documents/hocsau/Practice2
#       nohup bash practice2/run_experiments.sh > practice2/logs/master.log 2>&1 &
set -u
cd "$(dirname "$0")/.."   # về Practice2/
PY=myvenv/bin/python
T="practice2/train.py"

run () { echo "===== [$(date +%H:%M:%S)] START $* ====="; $PY $T "$@" && echo "===== DONE $* ====="; }

# Exercise 1: so sánh 3 kiến trúc (mode transfer, Adam 1e-3, bs32)
run --model resnet18    --mode transfer --epochs 6 --lr 1e-3 --batch-size 32 --run-name ex1_resnet18_transfer
run --model densenet121 --mode transfer --epochs 6 --lr 1e-3 --batch-size 32 --run-name ex1_densenet121_transfer
run --model vgg16       --mode transfer --epochs 4 --lr 1e-3 --batch-size 32 --run-name ex1_vgg16_transfer

# Transfer vs Fine-tune (ResNet18, SGD momentum)
run --model resnet18 --mode finetune --optimizer sgd --epochs 6 --lr 1e-2 --batch-size 32 --run-name ex_ft_resnet18_finetune

# Exercise 2: sweep hyperparameter trên ResNet18-transfer
run --model resnet18 --mode transfer --epochs 6 --lr 1e-4 --batch-size 32 --run-name ex2_lr1e-4
run --model resnet18 --mode transfer --epochs 6 --lr 1e-3 --batch-size 64 --run-name ex2_bs64
run --model resnet18 --mode transfer --optimizer sgd --epochs 6 --lr 1e-2 --batch-size 32 --run-name ex2_sgd

echo "===== [$(date +%H:%M:%S)] TẤT CẢ EXPERIMENTS XONG ====="
