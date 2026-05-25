from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "baseline.joblib"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File is not at {MODEL_PATH}. "
            "Run: python src/02_train_baseline.py"
        )
    return joblib.load(MODEL_PATH)


def predict_article(model, text: str):
    text = text.strip()
    if not text:
        raise ValueError("Article text is empty.")
    
    pred = model.predict([text])[0]
    
    proba = model.predict_proba([text])[0]
    confidence = float(proba[pred])
    
    label_name = "fake" if pred == 1 else "real"
    return int(pred), label_name, confidence


def main():
    model = load_model()
    print("Baseline fake-news classifier active.")
    print("Enter article text (blank line to finish). Type 'DONE' on its own line to run prediction.")
    print()
    
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "DONE":
            break
        lines.append(line)
        
    text = "\n".join(lines).strip()
    label, label_name, confidence = predict_article(model, text)
    
    print()
    print("=== Prediction ===")
    if(confidence < 0.70):
        print("Unclear prediction. Confidence is low. (Confidence: {confidence:.2%})")
    else:
        print(f"Label:      {label} ({label_name})")
        print(f"Confidence: {confidence:.2%}")
        print()
    

if __name__ == "__main__":
    main()