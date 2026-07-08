# -*- coding: utf-8 -*-
"""
Auto-label sentiment for anonymized Vietnamese feedback data with ViDeBERTa.

Input:
    CSV/XLSX from STEP2_Anonymize. The script prefers the
    "content_anonymized" column, then falls back to "content".

Output:
    Original data plus:
        - sentiment_label
        - sentiment_score
        - sentiment_raw_label
        - sentiment_raw_scores

Install:
    pip install transformers torch pandas openpyxl tqdm

Example:
    python "STEP3_Labeling/label_videberta.py" --input STEP2_Anonymize/posts_anonymized.csv --output STEP3_Labeling/output_labeled_posts_videberta.csv

    python "STEP3_Labeling/label_videberta.py" --input STEP2_Anonymize/comments_anonymized.csv --output STEP3_Labeling/output_labeled_comments_videberta.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


MODEL_NAME = "Fsoft-AIC/videberta-base"
DEFAULT_TEXT_COLUMNS = ("content_anonymized", "content")

# LABEL_0 = negative, LABEL_1 = neutral, LABEL_2 = positive.
LABEL_NORMALIZATION = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "NEG": "negative",
    "NEGATIVE": "negative",
    "NEU": "neutral",
    "NEUTRAL": "neutral",
    "POS": "positive",
    "POSITIVE": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}

RAW_LABEL_BY_NORMALIZED_LABEL = {
    "negative": "LABEL_0",
    "neutral": "LABEL_1",
    "positive": "LABEL_2",
}


def read_table(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported input file type: {suffix}. Use .csv, .xlsx, or .xls")


def write_table(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
        return

    raise ValueError(f"Unsupported output file type: {suffix}. Use .csv, .xlsx, or .xls")


def choose_text_column(df: pd.DataFrame, requested_column: str | None = None) -> str:
    if requested_column:
        if requested_column not in df.columns:
            raise ValueError(
                f"Text column '{requested_column}' not found. Available columns: {list(df.columns)}"
            )
        return requested_column

    for column in DEFAULT_TEXT_COLUMNS:
        if column in df.columns:
            return column

    raise ValueError(
        "Cannot find a text column. Expected one of "
        f"{DEFAULT_TEXT_COLUMNS}, or pass --text_col explicitly. "
        f"Available columns: {list(df.columns)}"
    )


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def normalize_label(label: str) -> str:
    return LABEL_NORMALIZATION.get(label, LABEL_NORMALIZATION.get(label.upper(), label))


def normalize_raw_label(label: str) -> str:
    normalized_label = normalize_label(label)
    return RAW_LABEL_BY_NORMALIZED_LABEL.get(normalized_label, label)


def build_sentiment_pipeline(model_name: str, device: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if device is None:
        device_id = 0 if torch.cuda.is_available() else -1
    else:
        normalized = device.lower()
        if normalized == "cpu":
            device_id = -1
        elif normalized.startswith("cuda"):
            device_id = 0
        else:
            raise ValueError("--device must be 'cpu' or 'cuda'")

    return pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
        top_k=None,
        truncation=True,
        max_length=256,
    )


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def normalize_score_item(item: dict) -> dict:
    return {
        **item,
        "label": normalize_raw_label(str(item["label"])),
        "score": float(item["score"]),
    }


def predict_batch(classifier, texts: list[str]) -> list[dict]:
    if not texts:
        return []

    outputs = classifier(texts)
    predictions = []

    for scores in outputs:
        # Transformers can return either:
        # - dict for one example
        # - list[dict] for multi-class scores
        # - list[list[dict]] for batched multi-class scores
        if isinstance(scores, dict):
            scores = [scores]
        elif scores and isinstance(scores[0], dict):
            scores = scores
        elif scores and isinstance(scores[0], list):
            scores = scores[0]
        else:
            raise TypeError(f"Unexpected pipeline output shape: {type(scores)!r}")

        normalized_scores = [normalize_score_item(item) for item in scores]
        normalized_scores = sorted(
            normalized_scores,
            key=lambda item: item["score"],
            reverse=True,
        )
        best = normalized_scores[0]
        raw_label = str(best["label"])

        predictions.append(
            {
                "sentiment_label": normalize_label(raw_label),
                "sentiment_score": float(best["score"]),
                "sentiment_raw_label": raw_label,
                "sentiment_raw_scores": json.dumps(normalized_scores, ensure_ascii=False),
            }
        )

    return predictions


def label_dataframe(
    df: pd.DataFrame,
    text_col: str,
    model_name: str = MODEL_NAME,
    batch_size: int = 16,
    min_len: int = 1,
    device: str | None = None,
) -> pd.DataFrame:
    classifier = build_sentiment_pipeline(model_name=model_name, device=device)

    result_df = df.copy()
    texts = [clean_text(value) for value in result_df[text_col].tolist()]

    predictions: list[dict] = []
    for batch in tqdm(
        list(batched(texts, batch_size)),
        desc="Labeling sentiment",
        unit="batch",
    ):
        valid_positions = [idx for idx, text in enumerate(batch) if len(text) >= min_len]
        valid_texts = [batch[idx] for idx in valid_positions]
        batch_predictions = [
            {
                "sentiment_label": "",
                "sentiment_score": 0.0,
                "sentiment_raw_label": "",
                "sentiment_raw_scores": "[]",
            }
            for _ in batch
        ]

        for local_idx, pred in zip(valid_positions, predict_batch(classifier, valid_texts)):
            batch_predictions[local_idx] = pred

        predictions.extend(batch_predictions)

    pred_df = pd.DataFrame(predictions)
    return pd.concat([result_df.reset_index(drop=True), pred_df], axis=1)


def run(
    input_path: str,
    output_path: str,
    text_col: str | None = None,
    model_name: str = MODEL_NAME,
    batch_size: int = 16,
    min_len: int = 1,
    device: str | None = None,
) -> pd.DataFrame:
    df = read_table(input_path)
    selected_text_col = choose_text_column(df, text_col)

    labeled_df = label_dataframe(
        df=df,
        text_col=selected_text_col,
        model_name=model_name,
        batch_size=batch_size,
        min_len=min_len,
        device=device,
    )
    write_table(labeled_df, output_path)
    return labeled_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label Vietnamese sentiment with ViDeBERTa."
    )
    parser.add_argument("--input", required=True, help="Input CSV/XLSX from STEP2_Anonymize")
    parser.add_argument("--output", required=True, help="Output CSV/XLSX with sentiment labels")
    parser.add_argument(
        "--text_col",
        default=None,
        help="Text column to label. Default: content_anonymized, then content",
    )
    parser.add_argument("--model", default=MODEL_NAME, help="Hugging Face model name")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--min_len", type=int, default=1)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Default: use CUDA if available, otherwise CPU",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        text_col=args.text_col,
        model_name=args.model,
        batch_size=args.batch_size,
        min_len=args.min_len,
        device=args.device,
    )
