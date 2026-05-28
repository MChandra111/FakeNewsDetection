"""
Classify articles with fine-tuned DistilBERT and show top token attributions.

Run from project root: python src/06_predict_bert.py
"""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert-fake-news"

MAX_LENGTH = 512
UNCERTAINTY_THRESHOLD = 0.70
TOP_N = 15


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model folder not found: {MODEL_DIR}\n"
            "Run: python src/05_train_bert.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = get_device()
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _encode(tokenizer, device, text: str):
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoded.pop("token_type_ids", None)
    return {key: value.to(device) for key, value in encoded.items()}


def _token_attributions(model, tokenizer, device, text: str, target_class: int):
    """
    Gradient x embedding attributions for one output class (0=real, 1=fake).
    Higher score => token pushed the model more toward that class.
    """
    encoded = _encode(tokenizer, device, text)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    emb_layer = model.distilbert.embeddings.word_embeddings
    embeds = emb_layer(input_ids)
    embeds = embeds.detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(inputs_embeds=embeds, attention_mask=attention_mask).logits
    logits[0, target_class].backward()

    # Per-token attribution (simplified; not full Integrated Gradients)
    attr = (embeds.grad * embeds).sum(dim=-1).squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    pairs = []
    for i, (tok, value) in enumerate(zip(tokens, attr.detach().cpu().tolist())):
        if attention_mask[0, i].item() == 0:
            continue
        if tok in ("[CLS]", "[SEP]", ""):
            continue
        pairs.append((tok, float(value)))

    return pairs


def explain_bert_article(model, tokenizer, device, text: str, top_n: int = TOP_N):
    """
    Top tokens that support fake vs real according to gradient attributions.
    """
    text = text.strip()
    if not text:
        raise ValueError("Article text is empty.")

    fake_pairs = _token_attributions(model, tokenizer, device, text, target_class=1)
    real_pairs = _token_attributions(model, tokenizer, device, text, target_class=0)

    toward_fake = sorted(fake_pairs, key=lambda x: x[1], reverse=True)[:top_n]
    toward_real = sorted(real_pairs, key=lambda x: x[1], reverse=True)[:top_n]

    pred_result = predict_article(model, tokenizer, device, text)

    return {
        **pred_result,
        "toward_fake": toward_fake,
        "toward_real": toward_real,
    }


def predict_article(model, tokenizer, device, text: str):
    text = text.strip()
    if not text:
        raise ValueError("Article text is empty.")

    encoded = _encode(tokenizer, device, text)

    with torch.no_grad():
        outputs = model(**encoded)

    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=0)

    pred = int(torch.argmax(probs).item())
    proba_real = float(probs[0].item())
    proba_fake = float(probs[1].item())
    confidence = float(probs[pred].item())
    label_name = "fake" if pred == 1 else "real"

    return {
        "pred": pred,
        "label_name": label_name,
        "confidence": confidence,
        "proba_real": proba_real,
        "proba_fake": proba_fake,
    }


def print_explanation(result: dict, top_n: int = TOP_N):
    print()
    print(f"=== Top {top_n} tokens supporting FAKE (in this article) ===")
    print("token | attribution score")
    for tok, score in result["toward_fake"]:
        print(f"  {tok!r}: {score:.4f}")
    print()
    print(f"=== Top {top_n} tokens supporting REAL (in this article) ===")
    for tok, score in result["toward_real"]:
        print(f"  {tok!r}: {score:.4f}")
    print()
    print(
        "Note: BERT uses subword tokens (e.g. '##ing'). "
        "Scores show what pushed toward each class, not proof of falsehood."
    )


def main():
    model, tokenizer, device = load_model_and_tokenizer()

    print(f"DistilBERT classifier loaded on {device}.")
    print("Paste article text, then type DONE on its own line.")
    print()

    lines = []
    while True:
        line = input()
        if line.strip().upper() == "DONE":
            break
        lines.append(line)

    text = "\n".join(lines).strip()
    result = explain_bert_article(model, tokenizer, device, text)

    print()
    print("=== Prediction (DistilBERT) ===")
    print(f"Real: {result['proba_real']:.2%}  |  Fake: {result['proba_fake']:.2%}")
    print(f"Chosen: {result['pred']} ({result['label_name']})")

    if result["confidence"] < 0.7:
        print()
        print(
            f"Unclear — confidence is below {UNCERTAINTY_THRESHOLD:.0%} "
        )
        print("Treat this as weak evidence, not a definitive verdict.")
    elif result["confidence"] < 0.85:
        print()
        print(f"Leans towards: {result['label_name']}")
    else:
        print(f"Strong pattern match toward: {result['label_name']}")

    print_explanation(result)


if __name__ == "__main__":
    main()
