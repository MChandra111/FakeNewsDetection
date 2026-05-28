from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_news.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert-fake-news"

USE_QUICK_RUN = False
QUICK_TRAIN_SIZE = 6000
QUICK_EVAL_SIZE = 1500


def pick_data_path() -> Path:
    return DATA_PATH

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
NUM_LABELS = 2
SEED = 42


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
    }
    

def main():
    set_seed(SEED)
    
    print("=== Loading Data ===")
    data_path_to_use = pick_data_path()
    print("Using data file:", data_path_to_use)
    df = pd.read_csv(data_path_to_use)
    df = df[["full_text", "label"]].dropna()
    df = df.rename(columns={"full_text": "text", "label": "labels"})
    df["labels"] = df["labels"].astype(int)
    
    
    if USE_QUICK_RUN:
        
        df = (
            df.groupby("labels", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), QUICK_TRAIN_SIZE // 2), random_state=SEED))
            .reset_index(drop=True)
        )
        print(f"QUICK RUN: using {len(df)} articles.")
        
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["labels"],
    )
    
    if USE_QUICK_RUN and len(test_df) > QUICK_EVAL_SIZE:
        test_df = test_df.sample(QUICK_EVAL_SIZE, random_state=SEED)
        
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)
    
    print("Train: ", len(train_ds), "Test: ", len(test_ds))
    print()
    
    print("=== Load Tokenizer And Model ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={0: "real", 1: "fake"},
        label2id={"real": 0, "fake": 1},
    )
    
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
    
    print("=== Tokenize (may take a few minutes) ===")
    
    train_ds = train_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    device_msg = "GPU" if torch.cuda.is_available() else "CPU"
    
    print(f"=== Training on {device_msg} ===")
    
    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "bert-training-checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        fp16=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_fake",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    print("Training complete.")
    
    print()
    print("=== Final evaluation on test split ===")
    print(trainer.evaluate())
    print()
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    
    print("=== Saved model + tokenizer ===")
    print(MODEL_DIR)

if __name__ == "__main__":
    main()