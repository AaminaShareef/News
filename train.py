"""
train.py
--------
Fine-tunes DistilBERT on the verified training records from train.json.
Run AFTER human review has set verified=true on records.

Doc section 5.1: all-MiniLM-L6-v2 as base model.
Doc section 6.2: only verified records used for fine-tuning.
Doc section 6.3: train.json → fine-tune script only. Never mix with operational JSON.

Multi-label classification: one article can belong to multiple industries.
Labels: the 10 industry IDs from industry_map.yaml.

Usage:
    python train.py                        # use only verified records
    python train.py --all                  # use all records (weak labels too)
    python train.py --min-verified 50      # require at least 50 verified records
"""

import json
import logging
import argparse
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Load and filter training records
# ---------------------------------------------------------------------------

def load_training_records(
    train_json_path: Path,
    verified_only: bool = True,
    min_verified: int = 0,
) -> tuple[list, list, list]:
    """
    Load records from train.json.
    Returns (train_records, val_records, test_records).
    """
    with open(train_json_path, encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    log.info("Total records in train.json: %d", len(records))

    if verified_only:
        verified = [r for r in records if r.get("verified")]
        log.info("Verified records: %d", len(verified))
        if len(verified) < min_verified:
            raise ValueError(
                f"Only {len(verified)} verified records found. "
                f"Need at least {min_verified}. "
                f"Review train.json and set verified=true for confirmed labels."
            )
        records = verified
    else:
        log.warning(
            "Using ALL records including unverified weak labels. "
            "Accuracy may be lower. Use --verified for production."
        )

    train = [r for r in records if r["split"] == "train"]
    val   = [r for r in records if r["split"] == "val"]
    test  = [r for r in records if r["split"] == "test"]

    log.info("Split: train=%d  val=%d  test=%d", len(train), len(val), len(test))
    return train, val, test


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset(records: list, industry_ids: list, tokenizer, max_length: int = 128):
    """
    Build a HuggingFace Dataset from training records.
    Multi-label binary encoding: [1, 0, 1, 0, ...] per record.
    """
    import torch
    from torch.utils.data import Dataset

    class IndustryDataset(Dataset):
        def __init__(self, records, industry_ids, tokenizer, max_length):
            self.records      = records
            self.industry_ids = industry_ids
            self.tokenizer    = tokenizer
            self.max_length   = max_length

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            record = self.records[idx]
            text   = record["text"]

            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Multi-label binary vector
            label_vec = [
                1.0 if ind_id in record.get("labels", []) else 0.0
                for ind_id in self.industry_ids
            ]

            return {
                "input_ids":      encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels":         torch.tensor(label_vec, dtype=torch.float),
            }

    return IndustryDataset(records, industry_ids, tokenizer, max_length)


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def fine_tune(
    train_records: list,
    val_records: list,
    industry_ids: list,
    model_output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
):
    """
    Fine-tune all-MiniLM-L6-v2 for multi-label industry classification.
    Doc section 5.1: DistilBERT family, 384-dim, ~40ms per article.
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
        from sklearn.metrics import f1_score, roc_auc_score
        import numpy as np
    except ImportError:
        raise ImportError(
            "Missing dependencies. Run:\n"
            "pip install transformers torch scikit-learn"
        )

    from config.settings import DISTILBERT_MODEL

    n_labels = len(industry_ids)
    log.info("Loading base model: %s", DISTILBERT_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL,
        num_labels=n_labels,
        problem_type="multi_label_classification",
    )

    train_dataset = build_dataset(train_records, industry_ids, tokenizer)
    val_dataset   = build_dataset(val_records,   industry_ids, tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))   # sigmoid
        preds = (probs >= 0.40).astype(int)  # same threshold as doc
        f1  = f1_score(labels, preds, average="micro", zero_division=0)
        try:
            auc = roc_auc_score(labels, probs, average="micro")
        except Exception:
            auc = 0.0
        return {"f1_micro": round(f1, 4), "auc_micro": round(auc, 4)}

    training_args = TrainingArguments(
        output_dir=str(model_output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        logging_dir=str(model_output_dir / "logs"),
        logging_steps=10,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    log.info("Starting fine-tuning — epochs=%d  batch=%d  lr=%s", epochs, batch_size, learning_rate)
    trainer.train()

    # Save final model
    model_output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

    # Save industry ID mapping alongside model
    label_map = {i: ind_id for i, ind_id in enumerate(industry_ids)}
    with open(model_output_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    log.info("Model saved → %s", model_output_dir)
    return trainer


# ---------------------------------------------------------------------------
# Evaluate on test set
# ---------------------------------------------------------------------------

def evaluate_test(trainer, test_records, industry_ids, tokenizer):
    from transformers import AutoTokenizer
    test_dataset = build_dataset(test_records, industry_ids, tokenizer)
    results = trainer.evaluate(test_dataset)
    log.info("Test set results: %s", results)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for industry classification")
    parser.add_argument("--all", action="store_true", help="Use all records, not just verified")
    parser.add_argument("--min-verified", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    from config.settings import TRAIN_JSON, MODELS_DIR, INDUSTRY_IDS

    if not TRAIN_JSON.exists():
        log.error("train.json not found. Run: python utils/collect_trainingdata.py first.")
        sys.exit(1)

    train_r, val_r, test_r = load_training_records(
        TRAIN_JSON,
        verified_only=not args.all,
        min_verified=args.min_verified,
    )

    if not train_r:
        log.error("No training records available.")
        sys.exit(1)

    model_dir = MODELS_DIR / "distilbert_industry"

    trainer = fine_tune(
        train_r, val_r, INDUSTRY_IDS, model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    if test_r:
        from transformers import AutoTokenizer
        from config.settings import DISTILBERT_MODEL
        tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL)
        evaluate_test(trainer, test_r, INDUSTRY_IDS, tokenizer)

    log.info("Training complete. Model at: %s", model_dir)
    log.info("Next step: python app.py  (Track B live pipeline)")