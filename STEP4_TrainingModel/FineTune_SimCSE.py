# -*- coding: utf-8 -*-
"""
Fine-tune SimCSE tiếng Việt (VoVanPhuc/sup-SimCSE-VietNamese-phobert-base)
cho bài toán phân loại cảm xúc 3 lớp: 0=Tiêu cực, 1=Trung lập, 2=Tích cực.

SimCSE này build trên nền PhoBERT nhưng được pretrain THÊM bằng
contrastive learning để embedding câu tốt hơn -> giả thuyết là sẽ cho
biểu diễn câu tốt hơn PhoBERT gốc cho bài toán cần hiểu NGHĨA TOÀN CÂU
(như sentiment) thay vì chỉ token-level.

Cấu trúc script GIỐNG HỆT train_phobert.py (cùng random_state, cùng
tỷ lệ split, cùng cách tách từ, cùng metric) - CHỈ KHÁC checkpoint
gốc - để đảm bảo so sánh công bằng, khác biệt kết quả (nếu có) đến từ
bản chất pretrained weights, không phải từ khác biệt cách xử lý dữ liệu.

QUAN TRỌNG: Chưa chạy thử thực tế (sandbox không có quyền truy cập
huggingface.co). Anh chạy trên máy có mạng.

Cài đặt:
pip install transformers torch datasets scikit-learn pandas underthesea --break-system-packages
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from datasets import Dataset
from underthesea import word_tokenize

MODEL_NAME = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"

ID2LABEL = {0: "tieu_cuc", 1: "trung_lap", 2: "tich_cuc"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# GIỐNG HỆT train_phobert.py - bắt buộc để so sánh công bằng
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


def preprocess_for_phobert(text: str) -> str:
    """
    SimCSE này build trên PhoBERT nên CŨNG cần văn bản đã tách từ,
    y hệt PhoBERT gốc (VD: 'sinh_viên').
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text
    try:
        return word_tokenize(text, format="text")
    except Exception:
        return text


def load_and_split(input_path: str, content_col: str, label_col: str = "sentiment_label"):
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    df = df[[content_col, label_col]].dropna().copy()
    df[label_col] = df[label_col].astype(int)

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


def run(input_path: str, output_dir: str, content_col: str = "content",
        label_col: str = "sentiment_label", epochs: int = 4,
        batch_size: int = 16, learning_rate: float = 2e-5):

    train_df, val_df, test_df = load_and_split(input_path, content_col, label_col)

    print("Đang tách từ (word segmentation)...")
    for d in (train_df, val_df, test_df):
        d["text_segmented"] = d[content_col].apply(preprocess_for_phobert)

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

    # Lưu ý: SimCSE checkpoint gốc không có sẵn classification head,
    # nên khi load bằng AutoModelForSequenceClassification, phần head
    # (lớp Linear cuối) sẽ được KHỞI TẠO NGẪU NHIÊN mới - đây là hành
    # vi ĐÚNG và bình thường (transformers sẽ in cảnh báo về việc này,
    # không phải lỗi). Backbone (phần encoder) vẫn giữ nguyên trọng số
    # đã pretrain bằng contrastive learning.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Bắt đầu fine-tune SimCSE...")
    trainer.train()

    print("\n=== Đánh giá trên tập TEST ===")
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

    parser = argparse.ArgumentParser(description="Fine-tune SimCSE (VN) cho phân loại cảm xúc 3 lớp")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="./simcse_sentiment_model")
    parser.add_argument("--content_col", default="content")
    parser.add_argument("--label_col", default="sentiment_label")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_dir=args.output_dir,
        content_col=args.content_col,
        label_col=args.label_col,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )