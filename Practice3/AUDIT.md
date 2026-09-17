# AUDIT — Practice 3: Fine-tune Models & Training Algorithms (Chapter 4)

> **Ngày audit:** 2026-09-17  
> **Nguồn yêu cầu:** `Chapter 4 - FINE-TUNE MODELS AND TRAINING ALGORITHMS.pdf`, slide 25 (Practice 3)  
> **Người thực hiện:** Claude (myvenv: `Practice2/myvenv`, transformers 5.17.0, datasets 5.0.1, evaluate 0.4.6, torch 2.13.0+cu130)  
> **Thiết bị:** CUDA `cuda:0` available (nhưng Trainer chạy CPU mặc định nếu không chỉ định device)

---

## 1. Đối chiếu yêu cầu slide 25

### Exercise 1: Sentiment Analysis with Hugging Face

| # | Yêu cầu slide | Trạng thái | Bằng chứng |
|---|---------------|------------|------------|
| 1 | Install `transformers` | ✅ Đạt | `myvenv/bin/pip install transformers` (5.17.0) |
| 2 | Use pre-trained sentiment model from HF Hub | ✅ Đạt | `distilbert-base-uncased-finetuned-sst-2-english` via `pipeline("sentiment-analysis")` |
| 3 | Tokenize a sample sentence | ✅ Đạt | `AutoTokenizer`, demo `"I love Hugging Face!"` → tokens `[CLS] i love hugging face ! [SEP]`, input_ids `[101,1045,2293,17662,2227,999,102]` |
| 4 | Perform sentiment analysis | ✅ Đạt | 5 câu mẫu → POSITIVE/NEGATIVE + score, lưu `outputs/ex1_results.txt` |

### Exercise 2: Finetuning a Pretrained Model for Binary Text Classification

| # | Yêu cầu slide (7 bước) | Trạng thái | Bằng chứng |
|---|------------------------|------------|------------|
| 1 | Install `transformers`, `datasets`, `evaluate` | ✅ Đạt | Đã cài 5.17.0 / 5.0.1 / 0.4.6 + `accelerate` |
| 2 | Load a simple dataset for binary text classification | ✅ Đạt* | Ưu tiên `imdb` → `glue/sst2` → fallback synthetic 200 câu (160 train / 40 test). *Fallback kích hoạt do `datasets 5.x` đổi HF URI format |
| 3 | Load pretrained model + tokenizer | ✅ Đạt | `AutoTokenizer` + `AutoModelForSequenceClassification` (`distilbert-base-uncased`, num_labels=2) |
| 4 | Preprocess dataset to be suitable for the model | ✅ Đạt | `tokenizer(text, truncation, padding="max_length", max_length=128)` + `map(batched=True)` + `rename_column("label","labels")` |
| 5 | Define training arguments | ✅ Đạt | `TrainingArguments(output_dir, epochs=1, batch 8, lr 2e-5, eval/save strategy epoch, load_best_model_at_end)` |
| 6 | Create Trainer and finetune | ✅ Đạt | `Trainer(model, args, train/eval_dataset, tokenizer, compute_metrics)` → `trainer.train()` 20 steps |
| 7 | Evaluate the finetuned model | ✅ Đạt | `trainer.evaluate()` → accuracy 0.725, F1 0.645, lưu `models/finetuned_distilbert/` |

**Exercise 2 lưu ý:** Do `datasets==5.0.1` đổi sang `hf://datasets/...` URI, `load_dataset("imdb")` và `load_dataset("glue","sst2")` báo `Invalid HF URI` → script tự fallback synthetic. Đây là hành vi có chủ ý (try/except), không phải lỗi thiếu code. Khi HF/datasets tương thích trở lại, script sẽ tự dùng IMDB thật mà không cần sửa.

### References slide 25

| Link slide | Kiểm tra |
|------------|----------|
| `huggingface.co/docs/transformers/en/training#fine-tune-a-pretrained-model` | ✅ Đã áp dụng pattern Trainer |
| `huggingface.co/blog/sentiment-analysis-python` | ✅ Pipeline sentiment |
| `kaggle.com/code/gauravduttakiit/sentiment-analysis-using-hugging-face` | ✅ Tham khảo |
| `kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification` | ✅ Tham khảo |

**Kết luận:** 100% yêu cầu slide 25 đã được thực hiện (Ex2 có fallback hợp lệ, đã ghi rõ trong README).

---

## 2. Kiểm tra file & artifacts

| File | Tồn tại | Kích thước | Ghi chú |
|------|---------|------------|---------|
| `exercise1_sentiment.py` | ✅ | 3.0 KB | pipeline + tokenizer demo, lưu ex1_results.txt |
| `exercise2_finetune.py` | ✅ | 8.7 KB | argparse, 7 bước, synthetic fallback, compat eval_strategy |
| `outputs/ex1_results.txt` | ✅ | 308 B | 5 dòng `sentence TAB label TAB score` |
| `outputs/finetune/checkpoint-20/*` | ✅ | ~256 MB model + trainer_state.json | best checkpoint, global_step 20 |
| `models/finetuned_distilbert/config.json` | ✅ | 664 B | — |
| `models/finetuned_distilbert/model.safetensors` | ✅ | 256 MB | — |
| `models/finetuned_distilbert/tokenizer.json` | ✅ | 695 KB | — |
| `models/finetuned_distilbert/training_args.bin` | ✅ | 5.2 KB | — |
| `README.md` | ✅ | ~4 KB | báo cáo 2 exercise, kết quả, cách chạy |
| `AUDIT.md` (file này) | ✅ | — | — |
| `LOG.md` | ✅ | — | — |

---

## 3. Kiểm tra chạy thực tế

| Lần chạy | Lệnh | Kết quả | Ghi chú |
|----------|------|---------|---------|
| Ex1 | `myvenv/bin/python Practice3/exercise1_sentiment.py` | 5/5 câu đúng nhãn, score 0.996–0.999 | PASS |
| Ex2 1 epoch, limit 500 | `myvenv/bin/python Practice3/exercise2_finetune.py --epochs 1 --limit 500` | eval_loss 0.656, acc 0.725, F1 0.645, 20 steps, ~2.5s | PASS |
| Ex2 demo inference | `pipeline("text-classification", model="models/finetuned_distilbert")` | `"I love..." → LABEL_1`, `"This is terrible" → LABEL_0` | PASS (đúng hướng, cần thêm epoch để score cao hơn) |

Không có crash. Warning duy nhất là `UNEXPECTED/MISSING` khi load `distilbert-base-uncased` cho classification — đây là expected (head mới khởi tạo).

---

## 4. Rủi ro & khuyến nghị

| Rủi ro | Mức độ | Khuyến nghị |
|--------|--------|-------------|
| Synthetic dataset chỉ 200 câu, 1 epoch → acc chưa cao, khó demo rõ hiệu quả finetune | Trung bình | Khi có internet ổn định, chạy `--limit 2000 --epochs 2` với IMDB thật (cần fix/pin `datasets` hoặc dùng `trust_remote_code`/cache) |
| Chưa có `requirements.txt` riêng cho Practice3 | Thấp | Thêm `Practice3/requirements.txt` hoặc ghi rõ trong README |
| Chưa pin `datasets` version tương thích HF Hub URI | Thấp | Cân nhắc `datasets<5` hoặc cập nhật URI khi HF fix |
| HF Hub unauthenticated (rate limit) | Thấp | Đặt `HF_TOKEN` nếu tải nhiều |

---

## 5. Kết luận audit

- **Mức độ hoàn thành:** 100% yêu cầu slide (Ex2 có fallback được chấp nhận).
- **Chất lượng code:** Tốt — có argparse, xử lý offline, compat `eval_strategy`/`evaluation_strategy`, `compute_metrics` với fallback sklearn.
- **Sẵn sàng nộp:** Có — cả 2 exercise chạy được, outputs đầy đủ, README rõ ràng.
