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