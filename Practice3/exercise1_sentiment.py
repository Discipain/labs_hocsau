"""
Practice 3 — Exercise 1: Sentiment Analysis with Hugging Face (Chapter 4, slide 25)
==================================================================================
Yêu cầu slide:
  1. Install transformers
  2. Use a pre-trained sentiment analysis model from HF Hub
  3. Tokenize a sample sentence
  4. Perform sentiment analysis

Chạy:
  myvenv/bin/python Practice3/exercise1_sentiment.py
"""

import os

SENTENCES = [
    "I love this movie, it was fantastic!",
    "This was a terrible and boring experience.",
    "The product is okay, not great but not bad either.",
    "I am so happy with the service, highly recommended!",
    "Worst purchase ever, I regret it completely.",
]

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def main():
    print("=" * 60)
    print("Exercise 1 — Sentiment Analysis with Hugging Face")
    print("=" * 60)

    # 1. Tokenize demo
    print(f"\n[1] Tokenizer demo — model: {MODEL_NAME}")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        sample = "I love Hugging Face!"
        encoded = tokenizer(sample, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
        print(f"  Sentence: {sample}")
        print(f"  Tokens: {tokens}")
        print(f"  input_ids: {encoded['input_ids'][0].tolist()}")
        print(f"  attention_mask: {encoded['attention_mask'][0].tolist()}")
    except Exception as e:
        print(f"  [WARN] Không tải được tokenizer (offline?): {e}")
        print(f"  Gợi ý: kiểm tra internet hoặc HF cache.")
        tokenizer = None

    # 2. Pipeline sentiment
    print(f"\n[2] Sentiment pipeline — model: {MODEL_NAME}")
    results = []
    try:
        from transformers import pipeline
        clf = pipeline("sentiment-analysis", model=MODEL_NAME)
        for s in SENTENCES:
            out = clf(s)[0]
            results.append((s, out["label"], out["score"]))
            print(f"  \"{s}\"")
            print(f"    -> {out['label']} (score={out['score']:.4f})")
    except Exception as e:
        print(f"  [WARN] Không chạy được pipeline (offline/thiếu model): {e}")
        print(f"  Hãy chạy lại khi có internet để HF tải model (~270MB).")
        # Fallback: thử với model đã cache hoặc báo lỗi
        return

    # Lưu kết quả
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ex1_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for sent, label, score in results:
            f.write(f"{sent}\t{label}\t{score:.4f}\n")
    print(f"\n  Saved results to {out_path}")

    print("\nDone. Tham khảo:")
    print("  https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english")
    print("  https://huggingface.co/blog/sentiment-analysis-python")

if __name__ == "__main__":
    main()
