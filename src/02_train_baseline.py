from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_news_political_balanced.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "baseline.joblib"

# Loading data

df = pd.read_csv(DATA_PATH)

X = df["full_text"]
y = df["label"]

print("=== Dataset Loaded ===")
print("Total Articles: ", len(df))
print("Label counts (0=real, 1=fake): ")
print(y.value_counts().sort_index())
print()

# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print("=== Split Sizes ===")
print("Train: ", len(X_train))
print("Test: ", len(X_test))
print()


# Model pipeline

model = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                max_features=20_000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                stop_words="english",
            ),
        ),
        (
            "clf",
            LogisticRegression(max_iter=1000),
        )
    ]
)

print("=== Training ===")
model.fit(X_train, y_train)
print("Training done.")
print()

y_pred = model.predict(X_test)

print("=== Classification Report (test set) ===")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["real (0)", "fake (1)"],
    )
)

print("=== Confusion Matrix ===")
print("Rows = true label, Columns = predicted label")
print(confusion_matrix(y_test, y_pred))
print()

# Saving model for later

MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("=== Model Saved ===")
print(MODEL_PATH)