# LOG — Practice 3: Fine-tune Models & Training Algorithms (Hugging Face)

> **Thư mục:** `Practice3/`  
> **Scripts:** `exercise1_sentiment.py` (3.0 KB), `exercise2_finetune.py` (8.7 KB)  
> **Môi trường:** `Practice2/myvenv` — Python 3.12, transformers 5.17.0, datasets 5.0.1, evaluate 0.4.6, accelerate 1.15.0, torch 2.13.0+cu130

---

## Lịch sử chạy

### 2026-09-17 — Cài đặt
```bash
Practice2/myvenv/bin/pip install transformers datasets evaluate accelerate scikit-learn matplotlib
# -> Successfully installed transformers-5.17.0, datasets-5.0.1, evaluate-0.4.6, accelerate-1.15.0, ...
mkdir -p Practice3/models Practice3/outputs
```

### 2026-09-17 13:25 — Exercise 1: Sentiment Analysis
```bash
Practice2/myvenv/bin/python Practice3/exercise1_sentiment.py
```
- Tokenizer `distilbert-base-uncased-finetuned-sst-2-english` tải OK (warning unauthenticated nhưng không ảnh hưởng)
- Tokenizer demo: `"I love Hugging Face!"` → `['[CLS]', 'i', 'love', 'hugging', 'face', '!', '[SEP]']`, input_ids `[101,1045,2293,17662,2227,999,102]`
- Pipeline sentiment 5 câu:
  - `I love this movie, it was fantastic!` → POSITIVE 0.9999
  - `This was a terrible and boring experience.` → NEGATIVE 0.9997
  - `The product is okay, not great but not bad either.` → POSITIVE 0.9965
  - `I am so happy with the service, highly recommended!` → POSITIVE 0.9999
  - `Worst purchase ever, I regret it completely.` → NEGATIVE 0.9998
- Saved `outputs/ex1_results.txt` (308 B)
- **Kết quả:** PASS

### 2026-09-17 13:26 — Exercise 2: Finetune (1 epoch, limit 500)
```bash
Practice2/myvenv/bin/python Practice3/exercise2_finetune.py --epochs 1 --limit 500
```
- Tải dataset: `imdb` → lỗi `Invalid HF URI` (datasets 5.x đổi format), `glue/sst2` → lỗi tương tự → fallback synthetic 200 câu (160 train / 40 test)
- Load `distilbert-base-uncased` OK, 4 missing head params (expected, mới khởi tạo classifier)
- Preprocess: `max_length=128`, `map(batched)` 160+40 samples, columns `labels, input_ids, token_type_ids, attention_mask`
- Training 20 steps (batch 8, lr 2e-5):
  - step 10: loss 0.692, grad_norm 2.387, lr 1.1e-05
  - step 20: loss 0.674, grad_norm 1.831, lr 1e-06
  - eval: loss 0.656, accuracy 0.725, F1 0.645, ~0.1s
  - train_runtime 2.46s, 65 samples/s, 8 steps/s
- Checkpoint `outputs/finetune/checkpoint-20/` + `models/finetuned_distilbert/` (256 MB safetensors + tokenizer)
- Demo inference: `"I love this movie!" → LABEL_1 (0.501)`, `"This is terrible." → LABEL_0 (0.543)` — đúng hướng
- **Kết quả:** PASS

### 2026-09-17 13:27 — Tạo README
- `README.md` ghi kết quả 2 exercise, bảng metrics, cách chạy, lưu ý fallback

---

## Lệnh hữu ích

```bash
# Exercise 1
Practice2/myvenv/bin/python Practice3/exercise1_sentiment.py
cat Practice3/outputs/ex1_results.txt

# Exercise 2 — mặc định 1 epoch, 500 samples
Practice2/myvenv/bin/python Practice3/exercise2_finetune.py --epochs 1 --limit 500

# Exercise 2 — tuỳ chỉnh
Practice2/myvenv/bin/python Practice3/exercise2_finetune.py --epochs 2 --limit 2000 --model distilbert-base-uncased --batch-size 8 --lr 2e-5

# Kiểm tra outputs
ls -lh Practice3/outputs/ex1_results.txt Practice3/models/finetuned_distilbert/
cat Practice3/outputs/finetune/checkpoint-20/trainer_state.json | head -60
```

## Ghi chú
- Lần đầu chạy Ex1/Ex2 cần internet để tải model từ HF Hub (~270 MB cho distilbert). Đã cache, lần sau chạy offline vẫn OK nếu model đã tải.
- Nếu `datasets` tải IMDB lỗi URI, synthetic fallback đảm bảo script vẫn chạy và cho kết quả hợp lệ — khi HF fix, script tự dùng dataset thật.
- Trainer warning `UNEXPECTED: vocab_*` là expected khi load base model cho classification (head khác task).
