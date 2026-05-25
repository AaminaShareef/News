"""
models/domain_sentiment_model.py  —  ProcureGuard
===================================================
STEP 2 of 2 in the training pipeline.

What this does
--------------
1. Reads  training_data/combined_training_data.json
   (produced by collect_trainingdata.py — Step 1).
2. Fine-tunes DistilBERT for TWO classification heads:
     • Industry classifier  — 14 supply-chain industries
     • Sentiment classifier — Negative / Neutral / Positive
3. Saves the trained model to  models/news_classifier.pt
4. At app startup  train_domain_sentiment_model()  loads that saved model.
   If no saved model exists yet, falls back to a rule-based predictor
   so the app still runs before training has happened.

Public API (used by app.py)
----------------------------
    domain_model, sentiment_model = train_domain_sentiment_model()
    result = predict(domain_model, sentiment_model, text)
    # → (industry_label, sentiment_label, industry_confidence, sentiment_confidence)

Training via CLI
----------------
    python train.py
    python train.py --epochs 4
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
COMBINED_JSON = Path("training_data/combined_training_data.json")
MODEL_SAVE    = Path("models/news_classifier.pt")

# ── Label sets — must match risk_engine.py INDUSTRY_BASE keys ─────────────────
INDUSTRY_LABELS = [
    "Oil & Gas", "Semiconductors", "Pharmaceuticals", "Automotive",
    "Chemicals", "Metals & Mining", "Aerospace & Defence", "Agriculture & Food",
    "Electronics", "Textiles & Apparel", "Retail & FMCG", "Construction",
    "Logistics & Shipping", "Energy & Utilities",
]
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

# ── Heuristic fallback patterns (used when no trained model exists yet) ────────
_INDUSTRY_PATTERNS: dict[str, list[str]] = {
    "Oil & Gas":            ["oil", "gas", "crude", "lng", "refinery", "petroleum", "opec", "pipeline"],
    "Semiconductors":       ["chip", "semiconductor", "wafer", "tsmc", "fab", "microchip", "silicon"],
    "Pharmaceuticals":      ["drug", "pharma", "medicine", "vaccine", "api", "clinical", "dosage"],
    "Automotive":           ["car", "vehicle", "auto", "ev", "battery", "toyota", "ford", "volkswagen"],
    "Chemicals":            ["chemical", "fertiliser", "fertilizer", "polymer", "resin", "compound"],
    "Metals & Mining":      ["steel", "copper", "lithium", "cobalt", "aluminium", "aluminum", "mine", "ore"],
    "Aerospace & Defence":  ["aircraft", "aerospace", "aviation", "defence", "defense", "titanium", "mro"],
    "Agriculture & Food":   ["grain", "wheat", "corn", "harvest", "crop", "food", "agriculture", "soy"],
    "Electronics":          ["pcb", "display", "oled", "circuit", "component", "consumer electronics"],
    "Textiles & Apparel":   ["cotton", "yarn", "garment", "textile", "fabric", "apparel", "fashion"],
    "Retail & FMCG":        ["retail", "fmcg", "consumer goods", "supermarket", "inventory", "shelf"],
    "Construction":         ["cement", "lumber", "concrete", "steel", "construction", "building material"],
    "Logistics & Shipping": ["port", "shipping", "container", "freight", "logistics", "trucking", "cargo"],
    "Energy & Utilities":   ["grid", "power", "electricity", "utility", "renewable", "solar", "wind"],
}
_NEGATIVE_WORDS = [
    "shortage", "disruption", "crisis", "halt", "shutdown", "delay", "strike",
    "ban", "sanction", "embargo", "collapse", "failure", "outage", "recall",
    "threat", "risk", "warn", "spike", "blockage", "congestion", "scarcity",
]
_POSITIVE_WORDS = [
    "recover", "resume", "resolve", "ease", "improve", "stabilise",
    "stabilize", "reopen", "restart", "abundant", "agreement", "deal",
    "ceasefire", "lift", "increase output",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Rule-based fallback (cold-start — runs before any training)
# ═══════════════════════════════════════════════════════════════════════════════

class _RuleBasedDomainModel:
    def predict(self, text: str) -> tuple[str, float]:
        t = text.lower()
        scores: dict[str, float] = {}
        for industry, keywords in _INDUSTRY_PATTERNS.items():
            hits = sum(1 for kw in keywords if kw in t)
            if hits:
                scores[industry] = min(0.40 + hits * 0.08, 0.90)
        if not scores:
            return "Not_Relevant", 0.0
        best = max(scores, key=scores.get)
        return best, round(scores[best], 3)


class _RuleBasedSentimentModel:
    def predict(self, text: str) -> tuple[str, float]:
        t   = text.lower()
        neg = sum(1 for w in _NEGATIVE_WORDS if w in t)
        pos = sum(1 for w in _POSITIVE_WORDS if w in t)
        if neg > pos: return "Negative", min(0.55 + neg * 0.05, 0.92)
        if pos > neg: return "Positive", min(0.55 + pos * 0.05, 0.88)
        return "Neutral", 0.52


# ═══════════════════════════════════════════════════════════════════════════════
# DistilBERT dual-head classifier
# ═══════════════════════════════════════════════════════════════════════════════

class _DistilBertClassifier:
    """
    Dual-head DistilBERT classifier.
    Head 1 → 14-class industry softmax
    Head 2 → 3-class  sentiment softmax
    Both share the same DistilBERT backbone and are trained jointly.
    """

    def __init__(self):
        self._model     = None
        self._tokenizer = None
        self._device    = None
        self._fitted    = False

    # ── Persist ───────────────────────────────────────────────────────────────

    def load(self, path: Path) -> bool:
        try:
            import torch
            from transformers import DistilBertTokenizerFast
            checkpoint      = torch.load(str(path), map_location="cpu")
            self._tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            self._model     = self._build_model()
            self._model.load_state_dict(checkpoint["model_state"])
            self._device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device).eval()
            self._fitted    = True
            log.info("Loaded model from %s", path)
            return True
        except Exception as exc:
            log.warning("Could not load model from %s: %s", path, exc)
            return False

    def save(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self._model.state_dict()}, str(path))
        log.info("Model saved → %s", path)

    def _build_model(self):
        import torch.nn as nn
        from transformers import DistilBertModel

        class DualHeadModel(nn.Module):
            def __init__(self, n_ind, n_sent):
                super().__init__()
                self.bert          = DistilBertModel.from_pretrained("distilbert-base-uncased")
                hidden             = self.bert.config.hidden_size   # 768
                self.dropout       = nn.Dropout(0.3)
                self.industry_head = nn.Linear(hidden, n_ind)
                self.sent_head     = nn.Linear(hidden, n_sent)

            def forward(self, input_ids, attention_mask):
                out     = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls_vec = self.dropout(out.last_hidden_state[:, 0, :])
                return self.industry_head(cls_vec), self.sent_head(cls_vec)

        return DualHeadModel(len(INDUSTRY_LABELS), len(SENTIMENT_LABELS))

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, records: list[dict], epochs: int = 3) -> None:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, Dataset
            from transformers import DistilBertTokenizerFast
            from torch.optim import AdamW
        except ImportError:
            log.error("torch / transformers not installed. Run: pip install torch transformers")
            return

        # Only use records whose labels are in our known sets
        valid = [
            r for r in records
            if r.get("industry")      in INDUSTRY_LABELS
            and r.get("sentiment_hint") in SENTIMENT_LABELS
            and r.get("text", "").strip()
        ]
        if len(valid) < 10:
            log.error(
                "Only %d valid training records (need ≥ 10). "
                "Run collect_trainingdata.py first.", len(valid)
            )
            return

        log.info("Training on %d records for %d epochs...", len(valid), epochs)

        tokenizer       = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self._tokenizer = tokenizer
        ind_to_idx      = {label: i for i, label in enumerate(INDUSTRY_LABELS)}
        sent_to_idx     = {label: i for i, label in enumerate(SENTIMENT_LABELS)}

        class NewsDataset(Dataset):
            def __init__(self, recs, tok):
                self.recs = recs
                self.tok  = tok
            def __len__(self):
                return len(self.recs)
            def __getitem__(self, idx):
                r   = self.recs[idx]
                enc = self.tok(
                    r["text"][:512],
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                return {
                    "input_ids":      enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "industry_label": torch.tensor(ind_to_idx[r["industry"]], dtype=torch.long),
                    "sent_label":     torch.tensor(sent_to_idx[r["sentiment_hint"]], dtype=torch.long),
                }

        random.shuffle(valid)
        loader   = DataLoader(NewsDataset(valid, tokenizer), batch_size=16, shuffle=True)
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        model    = self._build_model().to(device)
        opt      = AdamW(model.parameters(), lr=2e-5)
        loss_fn  = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch in loader:
                ids   = batch["input_ids"].to(device)
                mask  = batch["attention_mask"].to(device)
                i_lbl = batch["industry_label"].to(device)
                s_lbl = batch["sent_label"].to(device)
                opt.zero_grad()
                i_log, s_log = model(ids, mask)
                loss = loss_fn(i_log, i_lbl) + loss_fn(s_log, s_lbl)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            log.info("  Epoch %d/%d  loss: %.4f", epoch, epochs, total_loss / max(len(loader), 1))

        self._model  = model
        self._fitted = True
        log.info("Training complete.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> tuple[str, float, str, float]:
        if not self._fitted or self._model is None:
            ind,  ic = _RuleBasedDomainModel().predict(text)
            sent, sc = _RuleBasedSentimentModel().predict(text)
            return ind, ic, sent, sc
        try:
            import torch
            import torch.nn.functional as F
            enc  = self._tokenizer(
                text[:512], return_tensors="pt", truncation=True,
                max_length=128, padding="max_length",
            )
            ids  = enc["input_ids"].to(self._device)
            mask = enc["attention_mask"].to(self._device)
            self._model.eval()
            with torch.no_grad():
                i_log, s_log = self._model(ids, mask)
            ip   = F.softmax(i_log,  dim=-1).cpu().numpy()[0]
            sp   = F.softmax(s_log,  dim=-1).cpu().numpy()[0]
            ii   = int(np.argmax(ip))
            si   = int(np.argmax(sp))
            ind  = INDUSTRY_LABELS[ii] if float(ip[ii]) >= 0.20 else "Not_Relevant"
            return ind, round(float(ip[ii]), 3), SENTIMENT_LABELS[si], round(float(sp[si]), 3)
        except Exception as exc:
            log.error("Inference error: %s", exc)
            ind,  ic = _RuleBasedDomainModel().predict(text)
            sent, sc = _RuleBasedSentimentModel().predict(text)
            return ind, ic, sent, sc


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

_classifier: Optional[_DistilBertClassifier] = None


def train_domain_sentiment_model() -> tuple:
    """
    Called at app startup by app.py.

    Loads models/news_classifier.pt if it exists.
    Falls back to rule-based predictor if the model file is not there yet.

    Returns (domain_model, sentiment_model) — same object for both,
    matching the tuple interface app.py expects.
    """
    global _classifier
    classifier = _DistilBertClassifier()

    if MODEL_SAVE.exists():
        ok = classifier.load(MODEL_SAVE)
        if not ok:
            log.warning("Model load failed — using rule-based fallback.")
    else:
        log.warning(
            "No trained model at %s. Using rule-based fallback. "
            "Run: python collect_trainingdata.py  then  python train.py",
            MODEL_SAVE,
        )

    _classifier = classifier
    return classifier, classifier   # app.py unpacks as (domain_model, sentiment_model)


def predict(
    domain_model,
    sentiment_model,
    text: str,
) -> tuple[str, Optional[str], float, float]:
    """
    Classify one article text.

    Returns
    -------
    (industry_label, sentiment_label, industry_confidence, sentiment_confidence)
    industry_label = "Not_Relevant" when model confidence < 0.20.
    """
    if not text or not text.strip():
        return "Not_Relevant", None, 0.0, 0.0

    clf = domain_model if isinstance(domain_model, _DistilBertClassifier) else _classifier
    if clf is None:
        ind, ic  = _RuleBasedDomainModel().predict(text)
        sent, sc = _RuleBasedSentimentModel().predict(text)
        return ind, sent, ic, sc

    industry, ind_conf, sentiment, sent_conf = clf.predict(text)
    return industry, sentiment, ind_conf, sent_conf


def get_all_industries() -> list:
    """Return all supported industry names (used by /api/industries route)."""
    return sorted(INDUSTRY_LABELS)


def train_model(
    csv_path:  str = None,
    epochs:    int = 3,
    save_path: str = str(MODEL_SAVE),
    json_path: str = str(COMBINED_JSON),
) -> str:
    """
    Fine-tune DistilBERT on collected training data and save the model.
    Called by train.py.

    Parameters
    ----------
    csv_path  : ignored (kept for backward compat with old train.py signature)
    epochs    : number of training epochs
    save_path : where to save the .pt checkpoint
    json_path : path to combined_training_data.json (Step 1 output)

    Returns
    -------
    str — path where model was saved
    """
    save_path = Path(save_path)
    json_path = Path(json_path)

    log.info("=" * 55)
    log.info("  STEP 2 — DistilBERT Fine-tuning")
    log.info("  Data : %s", json_path)
    log.info("  Save : %s", save_path)
    log.info("=" * 55)

    if not json_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {json_path}\n"
            f"Run Step 1 first:  python collect_trainingdata.py"
        )

    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    log.info("Loaded %d records from %s", len(records), json_path)

    # Log industry distribution before training
    ind_dist: dict[str, int] = {}
    for r in records:
        ind = r.get("industry", "Unknown")
        ind_dist[ind] = ind_dist.get(ind, 0) + 1
    log.info("Industry distribution:")
    for ind, cnt in sorted(ind_dist.items(), key=lambda x: -x[1]):
        log.info("  %-30s  %d", ind, cnt)

    classifier = _DistilBertClassifier()
    classifier.fit(records, epochs=epochs)

    if not classifier._fitted:
        raise RuntimeError("Training failed — see logs above.")

    classifier.save(save_path)
    return str(save_path)