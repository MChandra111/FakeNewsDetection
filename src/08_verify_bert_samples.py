"""
Sanity-check: run the saved BERT model on known rows from the training CSV.
If these are wrong, the issue is model/training — not just your pasted articles.

Run: python src/08_verify_bert_samples.py
"""

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert-fake-news"
DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "all_news_political_balanced.csv",
    PROJECT_ROOT / "data" / "processed" / "all_news_more_real.csv",
    PROJECT_ROOT / "data" / "processed" / "all_news_extended.csv",
    PROJECT_ROOT / "data" / "processed" / "all_news.csv",
]
MAX_LENGTH = 512
N_PER_CLASS = 5
SEED = 42


def load_data_path():
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No processed CSV found.")


def predict_one(model, tokenizer, device, text: str):
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoded.pop("token_type_ids", None)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits[0]
    probs = torch.softmax(logits, dim=0)
    pred = int(torch.argmax(probs).item())
    return pred, float(probs[0].item()), float(probs[1].item())


def main():
    data_path = load_data_path()
    print("Data file:", data_path)
    print("Model dir:", MODEL_DIR)
    print()

    df = pd.read_csv(data_path)[["full_text", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    for label_value, label_name in [(0, "real"), (1, "fake")]:
        sample = df[df["label"] == label_value].sample(
            n=min(N_PER_CLASS, (df["label"] == label_value).sum()),
            random_state=SEED,
        )
        print(f"=== Known {label_name} rows (label={label_value}) ===")
        for i, row in sample.iterrows():
            pred, p_real, p_fake = predict_one(
                model, tokenizer, device, row["full_text"][:2000]
            )
            ok = pred == label_value
            correct += int(ok)
            total += 1
            status = "OK" if ok else "WRONG"
            print(
                f"  [{status}] true={label_name} pred={pred} "
                f"(real={p_real:.1%}, fake={p_fake:.1%}) | "
                f"{row['full_text'][:80]}..."
            )
        print()

    print(f"Sample accuracy: {correct}/{total} ({correct/total:.1%})")
    print()
    print("If many WRONG rows appear here, labels or training are mismatched.")
    print("If OK here but your pasted articles fail, it's domain shift.")


if __name__ == "__main__":
    main()
