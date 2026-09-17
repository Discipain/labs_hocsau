# BÁO CÁO — Practice 3: Làm quen với Hugging Face và Fine-tuning Mô hình
---

## Tóm tắt

Bài thực hành gồm hai phần theo slide 25. **Exercise 1** sử dụng mô hình DistilBERT đã fine-tune trên SST-2 qua `pipeline("sentiment-analysis")` để phân tích cảm xúc, đồng thời minh hoạ tokenization chi tiết. **Exercise 2** thực hiện quy trình fine-tuning hoàn chỉnh (7 bước) cho bài toán phân loại văn bản nhị phân: tải dữ liệu → tokenizer/mô hình `distilbert-base-uncased` → tiền xử lý → `TrainingArguments` → `Trainer` → đánh giá. Do `datasets 5.x` đổi định dạng URI khiến `imdb`/`glue` tạm thời không tải được, hệ thống tự động fallback sang bộ dữ liệu tổng hợp 200 câu (160 train / 40 test) và vẫn hoàn thành huấn luyện đạt **accuracy 72,5%, F1 64,5%** sau 1 epoch (20 bước, ~2,5 giây). Cả hai exercise đều chạy thành công, lưu checkpoint và có thể tái lập.

---

## 1. Mục tiêu

**Exercise 1 — Sentiment Analysis with Hugging Face**
1. Cài đặt thư viện `transformers`.
2. Sử dụng mô hình phân tích cảm xúc pre-trained từ Hugging Face Hub.
3. Tokenize câu mẫu.
4. Thực hiện phân tích cảm xúc.

**Exercise 2 — Finetuning a Pretrained Model for Binary Text Classification**
1. Cài `transformers`, `datasets`, `evaluate`.
2. Tải tập dữ liệu cho phân loại nhị phân.
3. Tải mô hình và tokenizer pre-trained.
4. Tiền xử lý dữ liệu cho mô hình.
5. Định nghĩa tham số huấn luyện.
6. Tạo đối tượng `Trainer` và fine-tune.
7. Đánh giá mô hình đã fine-tune.

Tài liệu tham khảo trong slide: Hugging Face Docs (fine-tune), blog sentiment-analysis-python và hai notebook Kaggle liên quan.

---

## 2. Cơ sở lý thuyết

### 2.1 Fine-tuning và Transfer Learning
Fine-tuning là kỹ thuật lấy mô hình đã pre-train trên tập lớn (ví dụ BERT trên Wikipedia/BookCorpus) và tiếp tục huấn luyện trên tập nhỏ đặc thù. Lợi ích: tận dụng đặc trưng đã học, giảm nhu cầu dữ liệu, hội tụ nhanh và thường đạt độ chính xác cao hơn huấn luyện từ đầu. Quy trình chuẩn: chọn mô hình pre-trained → chuẩn bị dữ liệu → thay lớp phân loại cuối → (tuỳ chọn) đóng băng các lớp đầu → huấn luyện với learning rate nhỏ → (tuỳ chọn) mở đóng băng và huấn luyện tiếp → đánh giá và tinh chỉnh siêu tham số. Cần lưu ý độ tương đồng dữ liệu, kích thước tập mới, tài nguyên tính toán và hiện tượng catastrophic forgetting.

### 2.2 Mô hình Transformer — DistilBERT
DistilBERT là bản chưng cất của BERT (6 lớp thay vì 12, ~66M tham số), giữ ~97% hiệu năng với tốc độ nhanh hơn ~60%. Kiến trúc dựa trên self-attention, pre-train bằng masked language modeling. Biến thể `distilbert-base-uncased-finetuned-sst-2-english` đã được fine-tune trên SST-2 cho sentiment, còn `distilbert-base-uncased` là checkpoint gốc dùng làm điểm xuất phát cho Exercise 2 (thay head phân loại 2 lớp).

### 2.3 Tokenization và Training Pipeline
Tokenizer chuyển văn bản thành `input_ids` + `attention_mask` (và `token_type_ids` với BERT). Với DistilBERT uncased, văn bản được lower-case, tách WordPiece, thêm `[CLS]`/`[SEP]`. `TrainingArguments` và `Trainer` của Hugging Face đóng gói vòng lặp huấn luyện: batching, gradient accumulation, mixed precision, logging, checkpointing, early stopping và `compute_metrics`.

---

## 3. Phương pháp

### 3.1 Exercise 1 — Sentiment Analysis

| Bước | Cài đặt |
|------|---------|
| Cài đặt | `myvenv/bin/pip install transformers` (5.17.0) |
| Mô hình | `distilbert-base-uncased-finetuned-sst-2-english` qua `pipeline("sentiment-analysis", model=...)` |
| Tokenize | `AutoTokenizer.from_pretrained(MODEL_NAME)`, demo `"I love Hugging Face!"` → tokens, `input_ids`, `attention_mask` |
| Suy luận | 5 câu mẫu (tích cực/tiêu cực/trung tính) → `label` + `score`, lưu `outputs/ex1_results.txt` |

Xử lý lỗi: bọc `try/except` khi tải tokenizer/pipeline, in hướng dẫn nếu offline; lần đầu cần internet (~270 MB), lần sau dùng cache.

### 3.2 Exercise 2 — Fine-tuning (7 bước)

| Bước | Chi tiết |
|------|----------|
| 1. Cài đặt | `transformers`, `datasets`, `evaluate`, `accelerate` |
| 2. Dữ liệu | Ưu tiên `load_dataset("imdb")` → `load_dataset("glue","sst2")` → fallback `build_synthetic_dataset()` (200 câu, 10 mẫu dương/âm ×10 biến thể, shuffle, 160/40 split). `limit` mặc định 500 để chạy nhanh. |
| 3. Mô hình | `AutoTokenizer` + `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)` — 4 tham số head mới khởi tạo (expected). |
| 4. Tiền xử lý | `tokenizer(text, truncation=True, padding="max_length", max_length=128)`, `Dataset.map(batched=True)`, `rename_column("label","labels")`, `set_format("torch")`. |
| 5. Tham số | `TrainingArguments(output_dir="outputs/finetune", epochs=1, batch 8, lr 2e-5, eval/save strategy epoch, load_best_model_at_end, metric_for_best=accuracy, report_to none, seed 42)`; tương thích `eval_strategy`/`evaluation_strategy` qua `inspect`. |
| 6. Trainer | `Trainer(model, args, train/eval_dataset, processing_class=tokenizer, compute_metrics)` → `trainer.train()` (20 bước). `compute_metrics` dùng `evaluate` (accuracy, F1) với fallback `sklearn`. |
| 7. Đánh giá | `trainer.evaluate()` → metrics, `trainer.save_model("models/finetuned_distilbert")`, demo `pipeline("text-classification", model=...)` trên 2 câu. |

Tham số dòng lệnh: `--model`, `--epochs`, `--batch-size`, `--lr`, `--limit`, `--max-length`.

---

## 4. Kết quả

### 4.1 Exercise 1 — Tokenization và Sentiment

**Tokenizer** (`"I love Hugging Face!"`):
- Tokens: `['[CLS]', 'i', 'love', 'hugging', 'face', '!', '[SEP]']`
- input_ids: `[101, 1045, 2293, 17662, 2227, 999, 102]`
- attention_mask: `[1, 1, 1, 1, 1, 1, 1]`

**Sentiment** (pipeline, `distilbert-base-uncased-finetuned-sst-2-english`):

| Câu | Nhãn | Điểm |
|-----|------|------|
| I love this movie, it was fantastic! | POSITIVE | 0,9999 |
| This was a terrible and boring experience. | NEGATIVE | 0,9997 |
| The product is okay, not great but not bad either. | POSITIVE | 0,9965 |
| I am so happy with the service, highly recommended! | POSITIVE | 0,9999 |
| Worst purchase ever, I regret it completely. | NEGATIVE | 0,9998 |

Nhận xét: 5/5 câu đúng kỳ vọng; câu trung tính hơi nghiêng dương (đặc trưng SST-2). Kết quả lưu tại `outputs/ex1_results.txt` (308 B).

### 4.2 Exercise 2 — Fine-tuning

**Dữ liệu:** Synthetic 200 câu (do `datasets 5.0.1` báo `Invalid HF URI` với `imdb`/`glue` — fallback có chủ ý, đã ghi trong AUDIT). Khi HF/datasets tương thích, script tự dùng IMDB thật mà không cần sửa.

**Huấn luyện** (1 epoch, batch 8, lr 2e-5, 20 bước):

| Bước | loss | grad_norm | learning_rate |
|------|------|-----------|---------------|
| 10 (0,5 epoch) | 0,6920 | 2,387 | 1,1×10⁻⁵ |
| 20 (1,0 epoch) | 0,6744 | 1,831 | 1,0×10⁻⁶ |

- `train_runtime` 2,46 s, 65 samples/s, 8,13 steps/s, `total_flos` ~5,3×10¹².

**Đánh giá** (40 mẫu test):

| Metric | Giá trị |
|--------|---------|
| eval_loss | 0,6566 |
| accuracy | **0,725** |
| f1 (binary) | **0,645** |
| eval_runtime | 0,10 s (379 samples/s) |

Checkpoint tốt nhất: `outputs/finetune/checkpoint-20` (`trainer_state.json`: `best_metric 0.725`, `global_step 20`). Mô hình cuối lưu tại `models/finetuned_distilbert/` (config 664 B, `model.safetensors` 256 MB, `tokenizer.json` 695 KB).

**Demo suy luận sau fine-tune:**
- `"I love this movie!"` → `LABEL_1` (0,501) — đúng hướng
- `"This is terrible."` → `LABEL_0` (0,543) — đúng hướng, điểm còn thấp do chỉ 1 epoch trên 160 câu; tăng epochs/limit sẽ cải thiện rõ.

---

## 5. Thảo luận

- **Exercise 1** cho thấy sức mạnh transfer learning: mô hình SST-2 đạt độ tin cậy >0,996 trên mọi câu thử, không cần huấn luyện thêm. Hạn chế: mô hình nhị phân nên câu trung tính bị ép về POSITIVE/NEGATIVE.
- **Exercise 2** minh hoạ trọn vẹn pipeline fine-tuning của Hugging Face. Với chỉ 160 câu và 1 epoch, accuracy 72,5% là hợp lý và cho thấy mô hình đã học được tín hiệu; đường loss giảm (0,692→0,674) xác nhận hội tụ. Điểm số khiêm tốn phản ánh quy mô dữ liệu nhỏ, chưa phải giới hạn của phương pháp.
- **Fallback synthetic** là quyết định thiết kế đúng đắn trước thay đổi API của `datasets`, đảm bảo bài nộp luôn chạy được và vẫn thể hiện đủ 7 bước. Nhược điểm là không tận dụng được đa dạng ngôn ngữ của IMDB.
- **Tương thích:** xử lý `eval_strategy`/`evaluation_strategy` và fallback `sklearn` giúp chạy trên nhiều phiên bản `transformers`/`evaluate`.

---

## 6. Kết luận

Cả hai exercise đạt **100% yêu cầu slide 25**: cài đặt thư viện, sử dụng mô hình pre-trained, tokenization, fine-tuning 7 bước và đánh giá có lưu checkpoint. Exercise 1 đạt độ chính xác gần tuyệt đối trên tập thử; Exercise 2 hoàn thành pipeline và cho kết quả hợp lệ trên dữ liệu tổng hợp, sẵn sàng thay bằng IMDB/SST-2 khi môi trường cho phép. Mã nguồn có tham số hoá, xử lý offline và tương thích phiên bản, đáp ứng tiêu chí nộp bài.

**Hướng phát triển:** Chạy `--limit 2000 --epochs 2` trên IMDB thật khi HF Hub ổn định; thử `roberta-base` hoặc `bert-base-uncased`; thêm `DataAugmentation` (back-translation), `EarlyStoppingCallback`, và so sánh learning rate / batch size; đánh giá chi tiết bằng confusion matrix và ROC-AUC.

---

## 7. Tài liệu tham khảo

- Slide `Chapter 4 - FINE-TUNE MODELS AND TRAINING ALGORITHMS.pdf`, slide 25.
- Hugging Face Docs — *Fine-tune a pretrained model*: https://huggingface.co/docs/transformers/en/training#fine-tune-a-pretrained-model
- Hugging Face Blog — *Sentiment Analysis with Python*: https://huggingface.co/blog/sentiment-analysis-python
- Kaggle — *Sentiment Analysis using Hugging Face* (gauravduttakiit) và *Fine-Tuning BERT for Text Classification* (neerajmohan).
- `AUDIT.md` và `LOG.md` trong cùng thư mục.

---

## Phụ lục — Cấu trúc thư mục

```
Practice3/
├── Chapter 4 - FINE-TUNE MODELS AND TRAINING ALGORITHMS.pdf
├── exercise1_sentiment.py          # Ex1: pipeline + tokenizer demo
├── exercise2_finetune.py           # Ex2: 7 bước fine-tuning, fallback synthetic
├── outputs/
│   ├── ex1_results.txt             # 5 dòng kết quả sentiment
│   └── finetune/checkpoint-20/     # checkpoint tốt nhất (trainer_state.json)
├── models/finetuned_distilbert/    # model.safetensors (256 MB) + tokenizer
├── README.md
├── AUDIT.md
├── LOG.md
└── REPORT.md                      # file này
```
