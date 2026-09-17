"""
Practice 3 — Exercise 2: Finetune a Pretrained Model for Binary Text Classification
====================================================================================
Yêu cầu slide (7 bước):
  1. Install transformers, datasets, evaluate
  2. Load a simple dataset for binary text classification
  3. Load a pretrained model + tokenizer
  4. Preprocess dataset
  5. Define TrainingArguments
  6. Create Trainer and finetune
  7. Evaluate

Chạy:
  myvenv/bin/python Practice3/exercise2_finetune.py
  myvenv/bin/python Practice3/exercise2_finetune.py --epochs 1 --limit 300 --model distilbert-base-uncased

Ghi chú: script tự fallback sang synthetic dataset nếu không tải được từ HF Hub (offline).
"""

import argparse
import os
import numpy as np

MODEL_DEFAULT = "distilbert-base-uncased"

def build_synthetic_dataset():
    """Fallback khi offline — 200 câu synthetic."""
    from datasets import Dataset, DatasetDict
    pos = [
        "I love this movie", "This is fantastic", "Great product", "Amazing experience",
        "Highly recommended", "Wonderful service", "Excellent quality", "I am so happy",
        "Best purchase ever", "Outstanding performance",
    ]
    neg = [
        "I hate this movie", "This is terrible", "Bad product", "Awful experience",
        "Not recommended", "Poor service", "Low quality", "I am disappointed",
        "Worst purchase ever", "Horrible performance",
    ]
    texts, labels = [], []
    for _ in range(10):
        for t in pos:
            texts.append(t + " " + np.random.choice(["really", "very", "so", "quite", ""]))
            labels.append(1)
        for t in neg:
            texts.append(t + " " + np.random.choice(["really", "very", "so", "quite", ""]))
            labels.append(0)
    # shuffle
    idx = np.random.permutation(len(texts))
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]
    n_train = 160
    train = Dataset.from_dict({"text": texts[:n_train], "label": labels[:n_train]})
    test  = Dataset.from_dict({"text": texts[n_train:], "label": labels[n_train:]})
    return DatasetDict({"train": train, "test": test})

def load_dataset_binary(limit):
    from datasets import load_dataset
    # Thử IMDB trước (binary), fallback SST2
    for name, cfg, text_col in [
        ("imdb", None, "text"),
        ("glue", "sst2", "sentence"),
    ]:
        try:
            print(f"  Đang tải dataset {name}" + (f" ({cfg})" if cfg else "") + " ...")
            if cfg:
                ds = load_dataset(name, cfg)
                # glue/sst2: train/validation/test, map về train/test
                train = ds["train"]
                test = ds["validation"]
            else:
                ds = load_dataset(name)
                train = ds["train"]
                test = ds["test"]
            # Chuẩn hoá tên cột về "text"
            if text_col != "text" and text_col in train.column_names:
                train = train.rename_column(text_col, "text")
                test = test.rename_column(text_col, "text")
            if limit:
                train = train.select(range(min(limit, len(train))))
                test = test.select(range(min(limit // 4, len(test))))
            print(f"  Loaded: train={len(train)}, test={len(test)}, cols={train.column_names}")
            from datasets import DatasetDict
            return DatasetDict({"train": train, "test": test})
        except Exception as e:
            print(f"  Không tải được {name}: {e}")
            continue

    print("  Dùng synthetic dataset (offline fallback).")
    return build_synthetic_dataset()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DEFAULT, help="HF model name")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--limit", type=int, default=500, help="Giới hạn train samples để chạy nhanh")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    print("=" * 60)
    print("Exercise 2 — Finetuning a Pretrained Model (Binary Classification)")
    print("=" * 60)
    print(f"Model: {args.model} | epochs={args.epochs} | limit={args.limit}")

    # 2. Dataset
    print("\n[2] Load dataset")
    ds = load_dataset_binary(args.limit if args.limit > 0 else None)
    print(f"  Final: train={len(ds['train'])}, test={len(ds['test'])}")

    # 3. Tokenizer + Model
    print(f"\n[3] Load tokenizer & model: {args.model}")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    except Exception as e:
        print(f"  [ERROR] Không tải được model: {e}")
        print(f"  Kiểm tra internet / HF cache. Thử lại sau.")
        return

    # 4. Preprocess
    print(f"\n[4] Preprocess (max_length={args.max_length})")
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_length)

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in ("label",)])
    # datasets Trainer expects "labels"
    if "label" in tokenized["train"].column_names:
        tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    print(f"  Columns: {tokenized['train'].column_names}")
    sample = tokenized["train"][0]
    print(f"  Sample keys: {list(sample.keys())[:6]}")

    # 5. TrainingArguments + 6. Trainer
    print("\n[5-6] Training")
    from transformers import TrainingArguments, Trainer
    import evaluate

    out_dir = os.path.join(os.path.dirname(__file__), "outputs", "finetune")
    os.makedirs(out_dir, exist_ok=True)

    # Load metric — fallback sklearn nếu offline
    try:
        acc_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        use_evaluate = True
    except Exception:
        use_evaluate = False
        print("  evaluate metrics không tải được, dùng sklearn fallback.")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        if use_evaluate:
            acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
            f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"]
        else:
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="binary")
        return {"accuracy": acc, "f1": f1}

    # Tương thích nhiều version transformers: eval_strategy vs evaluation_strategy
    import inspect
    sig = inspect.signature(TrainingArguments.__init__)
    has_eval_strategy = "eval_strategy" in sig.parameters
    has_evaluation_strategy = "evaluation_strategy" in sig.parameters

    ta_kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        seed=42,
    )
    if has_eval_strategy:
        ta_kwargs["eval_strategy"] = "epoch"
    elif has_evaluation_strategy:
        ta_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 7. Evaluate
    print("\n[7] Evaluate")
    metrics = trainer.evaluate()
    print(f"  Metrics: {metrics}")

    # Save
    final_dir = os.path.join(os.path.dirname(__file__), "models", "finetuned_distilbert")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    print(f"  Saved model to {final_dir}")

    # Demo inference với model finetuned
    print("\n  Demo inference (finetuned):")
    from transformers import pipeline
    try:
        clf = pipeline("text-classification", model=final_dir, tokenizer=args.model)
        for s in ["I love this movie!", "This is terrible."]:
            print(f"    \"{s}\" -> {clf(s)[0]}")
    except Exception as e:
        print(f"    (skip pipeline demo: {e})")

if __name__ == "__main__":
    main()
