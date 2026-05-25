from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "baseline.joblib"
TOP_N = 15


def load_model():
    return joblib.load(MODEL_PATH)


def explain_article(model, text: str, top_n: int = TOP_N):
    text = text.strip()
    if not text:
        raise ValueError("Article text is empty.")
    
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    
    X = tfidf.transform([text])
    
    coef = clf.coef_[0]
    
    
    dense = X.toarray().ravel()
    contributions = dense * coef
    
    feature_names = tfidf.get_feature_names_out()
    
    nonzero = np.where(dense > 0)[0]
    pairs = [
        (feature_names[i], contributions[i], dense[i], coef[i])
        for i in nonzero
    ]
    
    toward_fake = sorted(
        [p for p in pairs if p[1] > 0],
        key = lambda x: x[1],
        reverse=True,
    )[:top_n]
    
    toward_real = sorted(
        [p for p in pairs if p[1] < 0],
        key = lambda x: x[1],
    )[:top_n]
    
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    
    return {
        "pred": int(pred),
        "proba_fake": float(proba[1]),
        "proba_real": float(proba[0]),
        "toward_fake": toward_fake,
        "toward_real": toward_real,
    }
    

def main():
    model = load_model()
    print("Explain baseline predictions (top words/phrases).")
    print("paste article, then DONE.")
    print()
    
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "DONE":
            break
        lines.append(line)
        
    text = "\n".join(lines).strip()
    result = explain_article(model, text)
    
    label_name = "fake" if result["pred"] == 1 else "real"
    print()
    
    print("=== Prediction ===")
    print(f"Label: fake={result['proba_fake']:.2%}, real={result['proba_real']:.2%}")
    print(f"Chosen: {result['pred']} ({label_name})")
    print()
    
    print(f"=== Top {TOP_N} tokens pushing toward FAKE (in this article) ===")
    print("token | contribution | tfidf x weight breakdown")
    for name, contrib, tfidf_val, weight in result["toward_fake"]:
        print(f"  {name!r}: {contrib:.4f}  (tfidf={tfidf_val:.4f} × weight={weight:.4f})")
    print()
    
    print(f"=== Top {TOP_N} tokens pushing toward REAL (in this article) ===")
    
    
    for name, contrib, tfidf_val, weight in result["toward_real"]:
        print(f"  {name!r}: {contrib:.4f}  (tfidf={tfidf_val:.4f} × weight={weight:.4f})")



if __name__ == "__main__":
    main()