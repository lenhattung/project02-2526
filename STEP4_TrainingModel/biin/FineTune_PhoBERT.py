# -*- coding: utf-8 -*-
"""
Fine-tune PhoBERT (vinai/phobert-base) cho bài toán phân loại cảm xúc
3 lớp: 0=Tiêu cực, 1=Trung lập, 2=Tích cực.

Input: file csv/xlsx có 2 cột: content (văn bản) và sentiment_label (0/1/2)
       -> chính là output của bước gán nhãn (label_sentiment_phobert.py)
       sau khi đã audit/sửa tay.

QUAN TRỌNG: Chưa chạy thử thực tế (sandbox không có quyền truy cập
huggingface.co để tải model). Anh chạy trên máy có mạng + lý tưởng là
có GPU (fine-tune full trên CPU sẽ khá chậm với dataset lớn).

Cài đặt:
pip install transformers torch datasets scikit-learn pandas underthesea --break-system-packages
"""

import numpy as np
import pandas as pd
import torch
import sys
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback,
)
from datasets import Dataset
from underthesea import word_tokenize

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

MODEL_NAME = "vinai/phobert-base"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TOKENIZATION_INPUT = SCRIPT_DIR / "Tokenization" / "merch_datasets_student_sentiment_preprocessed.csv"
DEFAULT_CONTENT_COL = "content_preprocessed"
DEFAULT_LABEL_COL = "label"
WORD_SEGMENTED_COL = "word_segmented"
LOSS_LABELS_VI = {
    "train_loss": "Mất mát huấn luyện",
    "eval_loss": "Mất mát đánh giá",
}
LEGACY_TOKENIZED_COLS = (
    "content_phobert_tokenized",
    "content_bgem3_tokenized",
    "content_simcse_tokenized",
)

# ID <-> nhãn: CỐ ĐỊNH để dùng chung, so sánh công bằng giữa các script
ID2LABEL = {0: "tieu_cuc", 1: "trung_lap", 2: "tich_cuc"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Tỷ lệ chia & random_state CỐ ĐỊNH - dùng giống hệt ở cả 3 script
# (train_phobert.py, train_simcse.py, train_bge_m3.py) để đảm bảo
# 3 model được đánh giá trên ĐÚNG CÙNG 1 tập test, so sánh công bằng.
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # tính trên phần còn lại sau khi tách test


def preprocess_for_phobert(text: str) -> str:
    """PhoBERT cần văn bản đã tách từ (VD: 'sinh_viên')."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text
    try:
        return word_tokenize(text, format="text")
    except Exception:
        return text


def read_dataframe(input_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path, encoding="utf-8-sig")
    return pd.read_excel(input_path)


def write_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(output_path, index=False)


def tokenize_dataset_file(
    input_path: str | Path = DEFAULT_TOKENIZATION_INPUT,
    output_path: str | Path | None = None,
    content_col: str = DEFAULT_CONTENT_COL,
    word_segmented_col: str = WORD_SEGMENTED_COL,
) -> Path:
    """Read dataset, tokenize content_col with underthesea, and save a new column."""
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path

    df = read_dataframe(input_path)
    if content_col not in df.columns:
        raise ValueError(f"Missing column '{content_col}'. Available columns: {list(df.columns)}")

    df = df.drop(columns=list(LEGACY_TOKENIZED_COLS), errors="ignore")
    df[word_segmented_col] = df[content_col].apply(preprocess_for_phobert)
    write_dataframe(df, output_path)
    print(f"Saved word_segmented column '{word_segmented_col}' to: {output_path}")
    return output_path


def build_model_text(df: pd.DataFrame, content_col: str) -> pd.Series:
    if content_col == WORD_SEGMENTED_COL:
        return df[content_col]
    if content_col == DEFAULT_CONTENT_COL and WORD_SEGMENTED_COL in df.columns:
        word_segmented = df[WORD_SEGMENTED_COL].fillna("").astype(str).copy()
        missing_mask = word_segmented.str.strip().eq("")
        if missing_mask.any():
            word_segmented.loc[missing_mask] = df.loc[missing_mask, content_col].apply(preprocess_for_phobert)
        return word_segmented
    return df[content_col].apply(preprocess_for_phobert)


def load_and_split(input_path: str, content_col: str, label_col: str = DEFAULT_LABEL_COL):
    df = read_dataframe(input_path)

    missing_cols = [col for col in (content_col, label_col) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols}. Available columns: {list(df.columns)}")

    df = df.dropna(subset=[content_col, label_col]).copy()
    df[label_col] = df[label_col].astype(int)

    # Tách train/temp trước, rồi tách temp thành val/test - dùng
    # stratify để giữ đúng tỷ lệ 3 nhãn ở cả 3 tập con (quan trọng
    # với bài toán dễ lệch nhãn như cảm xúc)
    train_df, temp_df = train_test_split(
        df, test_size=(TEST_SIZE + VAL_SIZE), random_state=RANDOM_STATE,
        stratify=df[label_col],
    )
    relative_test_size = TEST_SIZE / (TEST_SIZE + VAL_SIZE)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_size, random_state=RANDOM_STATE,
        stratify=temp_df[label_col],
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def build_regularized_config(
    hidden_dropout: float,
    attention_dropout: float,
    classifier_dropout: float,
):
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    dropout_values = {
        "hidden_dropout_prob": hidden_dropout,
        "attention_probs_dropout_prob": attention_dropout,
        "classifier_dropout": classifier_dropout,
    }
    for attr, value in dropout_values.items():
        if hasattr(config, attr):
            setattr(config, attr, value)
    return config


def safe_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def epoch_bucket(value) -> int | None:
    epoch = safe_float(value)
    if epoch is None or epoch <= 0:
        return None
    return max(1, int(np.ceil(epoch - 1e-8)))


def trainer_epoch_loss_history(log_history: list[dict]) -> pd.DataFrame:
    train_losses: dict[int, list[float]] = {}
    eval_losses: dict[int, list[float]] = {}
    fallback_train_loss: dict[int, float] = {}
    seen_epochs: list[int] = []

    for row in log_history:
        epoch = epoch_bucket(row.get("epoch"))
        if epoch is None:
            continue
        seen_epochs.append(epoch)

        loss = safe_float(row.get("loss"))
        if loss is not None:
            train_losses.setdefault(epoch, []).append(loss)

        train_loss = safe_float(row.get("train_loss"))
        if train_loss is not None:
            fallback_train_loss[epoch] = train_loss

        eval_loss = safe_float(row.get("eval_loss"))
        if eval_loss is not None:
            eval_losses.setdefault(epoch, []).append(eval_loss)

    if not train_losses and fallback_train_loss:
        train_losses = {epoch: [loss] for epoch, loss in fallback_train_loss.items()}

    if not train_losses and not eval_losses:
        return pd.DataFrame()

    max_epoch = max(seen_epochs + list(train_losses.keys()) + list(eval_losses.keys()))
    rows = []
    for epoch in range(1, max_epoch + 1):
        row = {"epoch": epoch}
        if epoch in train_losses:
            row["train_loss"] = float(np.mean(train_losses[epoch]))
        if epoch in eval_losses:
            row["eval_loss"] = float(np.mean(eval_losses[epoch]))
        rows.append(row)
    return pd.DataFrame(rows)


def save_trainer_loss_artifacts(trainer: Trainer, output_dir: str | Path, title: str) -> None:
    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    log_history = trainer.state.log_history
    with (report_dir / "trainer_log_history.json").open("w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)
    if log_history:
        pd.DataFrame(log_history).to_csv(report_dir / "trainer_log_history.csv", index=False, encoding="utf-8-sig")

    loss_history = trainer_epoch_loss_history(log_history)
    if loss_history.empty:
        return
    loss_history.to_csv(report_dir / "loss_curve_points.csv", index=False, encoding="utf-8-sig")

    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for column in ("train_loss", "eval_loss"):
        if column not in loss_history.columns:
            continue
        points = loss_history[["epoch", column]].dropna()
        if not points.empty:
            ax.plot(points["epoch"], points[column], marker="o", label=LOSS_LABELS_VI.get(column, column))
    max_epoch = max(float(loss_history["epoch"].max()), 1.0)
    ax.set_xlim(left=0, right=max_epoch)
    if max_epoch <= 25:
        ax.set_xticks(range(0, int(np.ceil(max_epoch)) + 1))
    ax.set_title(title)
    ax.set_xlabel("Vòng huấn luyện")
    ax.set_ylabel("Giá trị mất mát")
    ax.grid(True, alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "loss_curve.png", dpi=200)
    plt.close(fig)


def run(input_path: str, output_dir: str, content_col: str = DEFAULT_CONTENT_COL,
        label_col: str = DEFAULT_LABEL_COL, epochs: int = 20,
        batch_size: int = 16, learning_rate: float = 2e-5,
        patience: int = 2, weight_decay: float = 1e-2,
        label_smoothing: float = 0.05, warmup_ratio: float = 0.1,
        lr_scheduler_type: str = "linear", monitor: str = "eval_loss",
        hidden_dropout: float = 0.2, attention_dropout: float = 0.2,
        classifier_dropout: float = 0.3):

    train_df, val_df, test_df = load_and_split(input_path, content_col, label_col)

    print("Đang chuẩn bị văn bản đã tách từ cho PhoBERT...")
    for d in (train_df, val_df, test_df):
        d["text_segmented"] = build_model_text(d, content_col)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    def tokenize_fn(batch):
        return tokenizer(batch["text_segmented"], truncation=True, max_length=256)

    def to_hf_dataset(df):
        ds = Dataset.from_pandas(
            df[["text_segmented", label_col]].rename(columns={label_col: "labels"}),
            preserve_index=False,
        )
        return ds.map(tokenize_fn, batched=True)

    train_ds = to_hf_dataset(train_df)
    val_ds = to_hf_dataset(val_df)
    test_ds = to_hf_dataset(test_df)

    model_config = build_regularized_config(hidden_dropout, attention_dropout, classifier_dropout)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        label_smoothing_factor=label_smoothing,
        lr_scheduler_type=lr_scheduler_type,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=monitor,
        greater_is_better=not monitor.endswith("loss"),
        logging_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    callbacks = []
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("Bắt đầu fine-tune PhoBERT...")
    trainer.train()
    save_trainer_loss_artifacts(trainer, output_dir, "Hàm mất mát PhoBERT")

    print("\n=== Đánh giá trên tập TEST (giữ nguyên, chưa từng thấy khi train) ===")
    test_results = trainer.predict(test_ds)
    preds = np.argmax(test_results.predictions, axis=-1)
    labels = test_results.label_ids
    print(classification_report(
        labels, preds, target_names=[ID2LABEL[i] for i in range(3)], digits=4
    ))

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nĐã lưu model vào: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT cho phân loại cảm xúc 3 lớp")
    parser.add_argument("--input", default=str(DEFAULT_TOKENIZATION_INPUT))
    parser.add_argument("--output_dir", default="./phobert_sentiment_model")
    parser.add_argument("--content_col", default=DEFAULT_CONTENT_COL)
    parser.add_argument("--label_col", default=DEFAULT_LABEL_COL)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--monitor", choices=["eval_loss", "eval_f1_macro"], default="eval_loss")
    parser.add_argument("--hidden_dropout", type=float, default=0.2)
    parser.add_argument("--attention_dropout", type=float, default=0.2)
    parser.add_argument("--classifier_dropout", type=float, default=0.3)
    parser.add_argument("--tokenize_only", action="store_true")
    parser.add_argument("--tokenized_output", default=None)
    parser.add_argument("--word_segmented_col", default=WORD_SEGMENTED_COL)
    args = parser.parse_args()

    if args.tokenize_only:
        tokenize_dataset_file(
            input_path=args.input,
            output_path=args.tokenized_output,
            content_col=args.content_col,
            word_segmented_col=args.word_segmented_col,
        )
    else:
        run(
            input_path=args.input,
            output_dir=args.output_dir,
            content_col=args.content_col,
            label_col=args.label_col,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            monitor=args.monitor,
            hidden_dropout=args.hidden_dropout,
            attention_dropout=args.attention_dropout,
            classifier_dropout=args.classifier_dropout,
        )
