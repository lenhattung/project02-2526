# -*- coding: utf-8 -*-
"""
Fine-tune BiLSTM cho bai toan sentiment/emotion 3 lop.

Script nay giu cung cach chia train/val/test voi FineTune_PhoBERT.py de
ket qua co the so sanh cong bang:
  - RANDOM_STATE = 42
  - TEST_SIZE = 0.15
  - VAL_SIZE = 0.15
  - stratify theo cot nhan

Mac dinh phu hop file dang dung:
  STEP3_Labeling/merch_datasets_student_sentiment.csv
  content_col = content_anonymized
  label_col   = label

Co the dung voi cac file cu hon bang cach truyen:
  --content_col content --label_col sentiment_label

Cai dat:
  pip install torch pandas scikit-learn tqdm underthesea

Vi du chay nhanh:
  python STEP4_TrainingModel/FineTune_LSTM.py

Vi du tuy bien:
  python STEP4_TrainingModel/FineTune_LSTM.py ^
    --input STEP3_Labeling/merch_datasets_student_sentiment.csv ^
    --output_dir STEP4_TrainingModel/lstm_sentiment_model ^
    --content_col content_anonymized ^
    --label_col label ^
    --epochs 10 ^
    --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "STEP3_Labeling" / "merch_datasets_student_sentiment.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().with_name("lstm_sentiment_model")

ID2LABEL = {0: "tieu_cuc", 1: "trung_lap", 2: "tich_cuc"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

LABEL_ALIASES = {
    "0": 0,
    "negative": 0,
    "neg": 0,
    "label_0": 0,
    "tieu_cuc": 0,
    "tiêu cực": 0,
    "tiêu_cực": 0,
    "1": 1,
    "neutral": 1,
    "neu": 1,
    "label_1": 1,
    "trung_lap": 1,
    "trung lập": 1,
    "trung_lập": 1,
    "2": 2,
    "positive": 2,
    "pos": 2,
    "label_2": 2,
    "tich_cuc": 2,
    "tích cực": 2,
    "tích_cực": 2,
}

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_table(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Chi ho tro CSV/XLSX, nhan duoc file: {path}")


def resolve_column(df: pd.DataFrame, requested: str, fallbacks: Iterable[str], kind: str) -> str:
    if requested in df.columns:
        return requested
    for col in fallbacks:
        if col in df.columns:
            print(f"[WARN] Khong thay cot {kind} '{requested}', dung fallback '{col}'.")
            return col
    available = ", ".join(df.columns.astype(str).tolist())
    raise ValueError(f"Khong tim thay cot {kind}. Can '{requested}' hoac {list(fallbacks)}. Cot hien co: {available}")


def normalize_label(value: object) -> int:
    if pd.isna(value):
        raise ValueError("Nhan bi thieu")

    if isinstance(value, (int, np.integer)):
        label_id = int(value)
        if label_id in ID2LABEL:
            return label_id

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        label_id = int(value)
        if label_id in ID2LABEL:
            return label_id

    key = str(value).strip().lower()
    if key.endswith(".0"):
        key = key[:-2]
    key = re.sub(r"\s+", " ", key)
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]

    raise ValueError(
        f"Nhan khong hop le: {value!r}. Hop le: 0/1/2, Negative/Neutral/Positive, LABEL_0/1/2."
    )


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def load_and_split(
    input_path: str | Path,
    content_col: str,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    df = read_table(input_path)
    content_col = resolve_column(df, content_col, ("content_anonymized", "content", "text"), "noi dung")
    label_col = resolve_column(df, label_col, ("label", "sentiment_label", "emotion"), "nhan")

    df = df[[content_col, label_col]].copy()
    df[content_col] = df[content_col].apply(clean_text)
    df = df[df[content_col].str.len() > 0].dropna(subset=[label_col]).copy()
    df["label_id"] = df[label_col].apply(normalize_label).astype(int)

    label_counts = df["label_id"].value_counts().sort_index()
    missing_labels = [label_id for label_id in ID2LABEL if label_id not in label_counts.index]
    if missing_labels:
        raise ValueError(f"Thieu nhan {missing_labels}; can du 3 lop de train/evaluate on dinh.")

    print("Phan bo nhan:")
    for label_id, count in label_counts.items():
        print(f"  {label_id} ({ID2LABEL[int(label_id)]}): {count}")

    train_df, temp_df = train_test_split(
        df,
        test_size=(TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_STATE,
        stratify=df["label_id"],
    )
    relative_test_size = TEST_SIZE / (TEST_SIZE + VAL_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=temp_df["label_id"],
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        content_col,
        label_col,
    )


class VietnameseTokenizer:
    def __init__(self, mode: str = "underthesea", lower: bool = True):
        self.mode = mode
        self.lower = lower
        self._word_tokenize = None

        if mode == "underthesea":
            try:
                from underthesea import word_tokenize

                self._word_tokenize = word_tokenize
            except Exception as exc:
                print(f"[WARN] Khong import duoc underthesea ({exc}). Fallback sang tokenizer simple.")
                self.mode = "simple"

    def __call__(self, text: str) -> list[str]:
        text = clean_text(text)
        if self.lower:
            text = text.lower()

        if self.mode == "underthesea" and self._word_tokenize is not None:
            try:
                segmented = self._word_tokenize(text, format="text")
                return [tok for tok in segmented.split() if tok]
            except Exception:
                pass

        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def build_vocab(tokenized_texts: Iterable[list[str]], max_vocab: int, min_freq: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, freq in counter.most_common(max(0, max_vocab - len(vocab))):
        if freq < min_freq:
            break
        vocab[token] = len(vocab)

    return vocab


def encode_tokens(tokens: list[str], vocab: dict[str, int], max_length: int) -> tuple[list[int], int]:
    ids = [vocab.get(token, UNK_ID) for token in tokens[:max_length]]
    length = max(1, len(ids))

    if not ids:
        ids = [UNK_ID]
    if len(ids) < max_length:
        ids.extend([PAD_ID] * (max_length - len(ids)))

    return ids, min(length, max_length)


class SentimentDataset(Dataset):
    def __init__(
        self,
        texts: Iterable[str],
        labels: Iterable[int],
        tokenizer: VietnameseTokenizer,
        vocab: dict[str, int],
        max_length: int,
    ):
        self.samples = []
        for text, label in zip(texts, labels):
            ids, length = encode_tokens(tokenizer(text), vocab, max_length)
            self.samples.append((ids, length, int(label), clean_text(text)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        ids, length, label, text = self.samples[index]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "text": text,
        }


class BiLSTMSentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        pad_id: int = PAD_ID,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(last_hidden))


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_class_weights(labels: Iterable[int], device: torch.device) -> torch.Tensor:
    counts = Counter(int(label) for label in labels)
    total = sum(counts.values())
    weights = [total / (len(ID2LABEL) * max(1, counts.get(label_id, 0))) for label_id in range(len(ID2LABEL))]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def cross_entropy_loss_parts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> tuple[float, float]:
    losses = nn.functional.cross_entropy(logits, labels, weight=weight, reduction="none")
    if weight is None:
        denominator = float(labels.numel())
    else:
        denominator = float(weight.detach()[labels].sum().detach().cpu().item())
    return float(losses.sum().detach().cpu().item()), denominator


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    loss_denom = 0.0
    unweighted_loss_sum = 0.0
    unweighted_loss_count = 0.0
    all_preds = []
    all_labels = []
    loss_weight = getattr(criterion, "weight", None)

    progress = tqdm(loader, desc="Train", leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["length"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        batch_loss_sum, batch_loss_denom = cross_entropy_loss_parts(logits, labels, loss_weight)
        batch_unweighted_sum, batch_unweighted_count = cross_entropy_loss_parts(logits, labels, None)
        loss_sum += batch_loss_sum
        loss_denom += batch_loss_denom
        unweighted_loss_sum += batch_unweighted_sum
        unweighted_loss_count += batch_unweighted_count
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        progress.set_postfix(loss=loss_sum / loss_denom if loss_denom else 0.0)

    return {
        "loss": loss_sum / loss_denom if loss_denom else 0.0,
        "unweighted_loss": unweighted_loss_sum / unweighted_loss_count if unweighted_loss_count else 0.0,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, object]:
    model.eval()
    loss_sum = 0.0
    loss_denom = 0.0
    unweighted_loss_sum = 0.0
    unweighted_loss_count = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_texts = []
    loss_weight = getattr(criterion, "weight", None)

    for batch in tqdm(loader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        lengths = batch["length"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        batch_loss_sum, batch_loss_denom = cross_entropy_loss_parts(logits, labels, loss_weight)
        batch_unweighted_sum, batch_unweighted_count = cross_entropy_loss_parts(logits, labels, None)
        loss_sum += batch_loss_sum
        loss_denom += batch_loss_denom
        unweighted_loss_sum += batch_unweighted_sum
        unweighted_loss_count += batch_unweighted_count
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_texts.extend(batch["text"])

    return {
        "loss": loss_sum / loss_denom if loss_denom else 0.0,
        "unweighted_loss": unweighted_loss_sum / unweighted_loss_count if unweighted_loss_count else 0.0,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
        "texts": all_texts,
    }


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def monitor_improved(current: float, best: float, monitor: str, min_delta: float) -> bool:
    if monitor.endswith("loss"):
        return current < best - min_delta
    return current > best + min_delta


def initial_monitor_score(monitor: str) -> float:
    return math.inf if monitor.endswith("loss") else -math.inf


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    vocab: dict[str, int],
    config: dict[str, object],
    best_info: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "config": config,
        "id2label": ID2LABEL,
        **best_info,
    }
    torch.save(checkpoint, output_dir / "lstm_model.pt")
    save_json(output_dir / "vocab.json", vocab)
    save_json(output_dir / "label_mapping.json", {"id2label": ID2LABEL, "label2id": LABEL2ID})


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def run(
    input_path: str | Path,
    output_dir: str | Path,
    content_col: str = "content_anonymized",
    label_col: str = "label",
    epochs: int = 50,
    min_epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 5e-4,
    embedding_dim: int = 128,
    hidden_size: int = 128,
    num_layers: int = 1,
    dropout: float = 0.5,
    max_length: int = 128,
    max_vocab: int = 30000,
    min_freq: int = 2,
    tokenizer_mode: str = "underthesea",
    lower: bool = True,
    patience: int = 5,
    monitor: str = "val_loss",
    min_delta: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    class_weight: bool = True,
    num_workers: int = 0,
    device_arg: str = "auto",
) -> None:
    set_seed(RANDOM_STATE)
    output_dir = Path(output_dir)
    device = resolve_device(device_arg)
    min_epochs = max(1, int(min_epochs))
    if epochs < min_epochs:
        print(f"[INFO] epochs={epochs} < min_epochs={min_epochs}; tu dong nang epochs len {min_epochs}.")
        epochs = min_epochs

    train_df, val_df, test_df, text_col, resolved_label_col = load_and_split(input_path, content_col, label_col)

    tokenizer = VietnameseTokenizer(mode=tokenizer_mode, lower=lower)
    print(f"Dung tokenizer: {tokenizer.mode} | lower={lower}")

    print("Dang build vocabulary tu tap train...")
    train_tokens = [tokenizer(text) for text in train_df[text_col].tolist()]
    vocab = build_vocab(train_tokens, max_vocab=max_vocab, min_freq=min_freq)
    print(f"Vocab size: {len(vocab)}")

    train_ds = SentimentDataset(train_df[text_col].tolist(), train_df["label_id"].tolist(), tokenizer, vocab, max_length)
    val_ds = SentimentDataset(val_df[text_col].tolist(), val_df["label_id"].tolist(), tokenizer, vocab, max_length)
    test_ds = SentimentDataset(test_df[text_col].tolist(), test_df["label_id"].tolist(), tokenizer, vocab, max_length)

    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = make_loader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model_config = {
        "vocab_size": len(vocab),
        "embedding_dim": embedding_dim,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": len(ID2LABEL),
        "dropout": dropout,
        "pad_id": PAD_ID,
    }
    model = BiLSTMSentimentClassifier(**model_config).to(device)

    weights = compute_class_weights(train_df["label_id"].tolist(), device) if class_weight else None
    if weights is not None:
        print(f"Class weights: {[round(float(w), 4) for w in weights.detach().cpu().tolist()]}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    run_config = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "content_col": text_col,
        "label_col": resolved_label_col,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "val_size": VAL_SIZE,
        "epochs": epochs,
        "min_epochs": min_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "min_freq": min_freq,
        "max_vocab": max_vocab,
        "class_weight": class_weight,
        "max_length": max_length,
        "lower": lower,
        "tokenizer_mode": tokenizer.mode,
        "patience": patience,
        "monitor": monitor,
        "min_delta": min_delta,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        **model_config,
    }
    save_json(output_dir / "training_config.json", run_config)

    best_monitor_score = initial_monitor_score(monitor)
    max_val_f1 = -1.0
    best_checkpoint_val_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    history = []

    print(f"Bat dau train BiLSTM tren device: {device}")
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip)
        val_metrics = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_unweighted_loss": train_metrics["unweighted_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_unweighted_loss": val_metrics["unweighted_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={row['train_loss']:.4f} train_f1={row['train_f1_macro']:.4f} | "
            f"val_loss={row['val_loss']:.4f} val_f1={row['val_f1_macro']:.4f}"
        )

        max_val_f1 = max(max_val_f1, row["val_f1_macro"])
        current_monitor_score = float(row[monitor])
        if monitor_improved(current_monitor_score, best_monitor_score, monitor, min_delta):
            best_monitor_score = current_monitor_score
            best_checkpoint_val_f1 = row["val_f1_macro"]
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                output_dir,
                model,
                vocab,
                run_config,
                {
                    "best_epoch": best_epoch,
                    "best_monitor": monitor,
                    "best_monitor_score": best_monitor_score,
                    "best_val_f1_macro": best_checkpoint_val_f1,
                    "max_val_f1_macro": max_val_f1,
                    "best_val_loss": row["val_loss"],
                    "best_val_unweighted_loss": row["val_unweighted_loss"],
                },
            )
            print(f"  -> Luu best model moi vao: {output_dir / 'lstm_model.pt'} ({monitor}={best_monitor_score:.4f})")
        else:
            # Khong tinh patience truoc/sat min_epochs; chi bat dau dem tu epoch > min_epochs.
            if epoch > min_epochs:
                bad_epochs += 1
                if patience > 0 and bad_epochs >= patience:
                    print(
                        f"Early stopping sau {bad_epochs} epoch khong cai thien {monitor}, "
                        f"sau khi da train toi thieu {min_epochs} epoch."
                    )
                    break
            else:
                bad_epochs = 0

    save_json(output_dir / "history.json", history)
    save_json(output_dir / "history_best_loss.json", [row for row in history if row["epoch"] <= best_epoch])

    checkpoint = torch.load(output_dir / "lstm_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        f"\n=== Danh gia tren tap TEST - best epoch {best_epoch}, "
        f"{monitor}={best_monitor_score:.4f}, val_f1={best_checkpoint_val_f1:.4f} ==="
    )
    test_metrics = evaluate(model, test_loader, criterion, device)
    report = classification_report(
        test_metrics["labels"],
        test_metrics["preds"],
        target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))],
        digits=4,
        zero_division=0,
    )
    print(report)

    metrics_payload = {
        "best_epoch": best_epoch,
        "best_monitor": monitor,
        "best_monitor_score": best_monitor_score,
        "best_val_f1_macro": best_checkpoint_val_f1,
        "max_val_f1_macro": max_val_f1,
        "test_loss": test_metrics["loss"],
        "test_unweighted_loss": test_metrics["unweighted_loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "classification_report": report,
    }
    save_json(output_dir / "test_metrics.json", metrics_payload)

    pred_df = pd.DataFrame(
        {
            "text": test_metrics["texts"],
            "true_label": test_metrics["labels"],
            "true_name": [ID2LABEL[int(label)] for label in test_metrics["labels"]],
            "pred_label": test_metrics["preds"],
            "pred_name": [ID2LABEL[int(label)] for label in test_metrics["preds"]],
            "prob_negative": [float(row[0]) for row in test_metrics["probs"]],
            "prob_neutral": [float(row[1]) for row in test_metrics["probs"]],
            "prob_positive": [float(row[2]) for row in test_metrics["probs"]],
        }
    )
    pred_path = output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print(f"\nDa luu model: {output_dir / 'lstm_model.pt'}")
    print(f"Da luu metrics: {output_dir / 'test_metrics.json'}")
    print(f"Da luu du doan test: {pred_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BiLSTM cho sentiment/emotion 3 lop")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--content_col", default="content_anonymized")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_vocab", type=int, default=30000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--tokenizer", choices=["underthesea", "simple"], default="underthesea")
    parser.add_argument("--no_lower", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_epochs", type=int, default=20)
    parser.add_argument("--monitor", choices=["val_loss", "val_unweighted_loss", "val_f1_macro"], default="val_loss")
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_class_weight", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_path=args.input,
        output_dir=args.output_dir,
        content_col=args.content_col,
        label_col=args.label_col,
        epochs=args.epochs,
        min_epochs=args.min_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_length=args.max_length,
        max_vocab=args.max_vocab,
        min_freq=args.min_freq,
        tokenizer_mode=args.tokenizer,
        lower=not args.no_lower,
        patience=args.patience,
        monitor=args.monitor,
        min_delta=args.min_delta,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        class_weight=not args.no_class_weight,
        num_workers=args.num_workers,
        device_arg=args.device,
    )
