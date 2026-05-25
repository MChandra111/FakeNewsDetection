from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FAKE_PATH = RAW_DIR / "Fake.csv"
TRUE_PATH = RAW_DIR / "True.csv"
OUTPUT_PATH = PROCESSED_DIR / "all_news.csv"


fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

print("=== Raw file shapes (rows, columns) ===")
print("Fake.csv: ", fake_df.shape)
print("True.csv: ", true_df.shape)
print()

print("=== Column names ===")
print("Fake columns: ", list(fake_df.columns))
print("True columns: ", list(true_df.columns))
print()


fake_df['label'] = 1
true_df['label'] = 0

fake_df['label_name'] = "fake"
true_df['label_name'] = "true"


all_df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)

print("=== Combined dataset ===")
print("Total rows:", len(all_df))
print()
print("=== Label counts ===")
print(all_df["label_name"].value_counts())
print()


all_df["title"] = all_df["title"].fillna("")
all_df["text"] = all_df["text"].fillna("")

all_df["full_text"] = (
    all_df["title"].astype(str) + " " + all_df["text"].astype(str)
).str.strip()


empty_count = (all_df["full_text"].str.len() == 0).sum()
print("=== Empty articles (full_text length 0) ===")
print(empty_count)
print()

all_df = all_df[all_df["full_text"].str.len() > 0].copy()

print("=== After removing empty rows ===")
print("Total rows:", len(all_df))
print(all_df["label_name"].value_counts())
print()

all_df["text_length"] = all_df["full_text"].str.len()

print("=== Text length (characters) — describe() ===")
print(all_df.groupby("label_name")["text_length"].describe())
print()

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
all_df.to_csv(OUTPUT_PATH, index=False)
print("=== Saved ===")
print(OUTPUT_PATH)