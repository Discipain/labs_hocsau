# Practice 3 — Get Started with Hugging Face (Chapter 4)

## Yêu cầu (slide 25)

### Exercise 1: Sentiment Analysis with Hugging Face
1. Install `transformers`
2. Dùng pre-trained sentiment model từ Hugging Face Hub
3. Tokenize sample sentence
4. Perform sentiment analysis

### Exercise 2: Finetuning a Pretrained Model for Binary Text Classification
1. Install `transformers`, `datasets`, `evaluate`
2. Load dataset cho binary text classification
3. Load pretrained model + tokenizer
4. Preprocess dataset
5. Define TrainingArguments
6. Create Trainer & finetune
7. Evaluate

---

## Cài đặt

```bash
/home/toi/Documents/hocsau/Practice2/myvenv/bin/pip install transformers datasets evaluate accelerate scikit-learn matplotlib
```

## Chạy

```bash
# Exercise 1 — Sentiment Analysis
myvenv/bin/python Practice3/exercise1_sentiment.py

# Exercise 2 — Finetune (1 epoch, ~2 phút trên CPU)
/myvenv/bin/python Practice3/exercise2_finetune.py --epochs 1 --limit 500
# Tuỳ chọn: --model distilbert-base-uncased --batch-size 8 --lr 2e-5 --max-length 128
```

## Kết quả thực chạy

### Exercise 1 — Sentiment Analysis
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english` (DistilBERT finetuned trên SST-2)
- **Tokenizer demo** (`"I love Hugging Face!"`):
  - Tokens: `['[CLS]', 'i', 'love', 'hugging', 'face', '!', '[SEP]']`
  - input_ids: `[101, 1045, 2293, 17662, 2227, 999, 102]`

| Câu | Kết quả |
|-----|---------|
| "I love this movie, it was fantastic!" | POSITIVE 0.9999 |
| "This was a terrible and boring experience." | NEGATIVE 0.9997 |
| "The product is okay, not great but not bad either." | POSITIVE 0.9965 |
| "I am so happy with the service, highly recommended!" | POSITIVE 0.9999 |
| "Worst purchase ever, I regret it completely." | NEGATIVE 0.9998 |

- Output lưu tại `outputs/ex1_results.txt`

### Exercise 2 — Finetune
- **Model**: `distilbert-base-uncased` (66M params, 6 layers)
- **Dataset**: Synthetic 200 câu (fallback khi HF Hub không tải được `imdb`/`glue` do thay đổi URI format ở `datasets==5.x`) — 160 train / 40 test. Khi có internet ổn định, script sẽ tự dùng `imdb` (500 mẫu) hoặc `glue/sst2`.
- **Tokenization**: `max_length=128`, `truncation` + `padding=max_length`
- **Training**: 1 epoch, batch 8, lr 2e-5, AdamW (mặc định của Trainer)

| Metric | Giá trị |
|--------|---------|
| eval_loss | 0.656 |
| accuracy | 0.725 |
| f1 (binary) | 0.645 |

- Model finetuned lưu tại `models/finetuned_distilbert/` (có thể load lại bằng `pipeline("text-classification", model="...")`)
- Demo inference sau finetune: `"I love this movie!" → LABEL_1 (0.50)`, `"This is terrible." → LABEL_0 (0.54)` — với 1 epoch trên 160 câu, phân biệt đã đúng hướng; tăng epochs/limit sẽ cải thiện rõ rệt.

## Cấu trúc file
```
Practice3/
├── exercise1_sentiment.py      # Ex1: pipeline + tokenizer demo
├── exercise2_finetune.py       # Ex2: datasets → tokenize → Trainer
├── outputs/
│   ├── ex1_results.txt
│   └── finetune/               # checkpoints Trainer
├── models/finetuned_distilbert/
└── README.md
```

## Tham khảo (từ slide)
- https://huggingface.co/docs/transformers/en/training#fine-tune-a-pretrained-model
- https://huggingface.co/blog/sentiment-analysis-python
- https://www.kaggle.com/code/gauravduttakiit/sentiment-analysis-using-hugging-face
- https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification
