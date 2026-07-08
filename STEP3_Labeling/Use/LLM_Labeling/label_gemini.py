# -*- coding: utf-8 -*-
"""
Auto-label sentiment for anonymized Vietnamese feedback data with Gemini.

Model:
    gemini-3.5-flash

Input:
    CSV/XLSX from STEP2_Anonymize. The script prefers the
    "content_anonymized" column, then falls back to "content".

Output:
    Original data plus:
        - sentiment_label
        - sentiment_score
        - sentiment_raw_label
        - sentiment_raw_scores
        - status

Install:
    pip install google-genai pandas openpyxl tqdm

Environment:
    Set GEMINI_API_KEY or GOOGLE_API_KEY before running.

Example:
    python STEP3_Labeling/LLM_Labeling/label_gemini.py --input STEP2_Anonymize/posts_anonymized.csv --output STEP3_Labeling/LLM_Labeling/output_labeled_posts_gemini.csv

    python STEP3_Labeling/LLM_Labeling/label_gemini.py --input STEP2_Anonymize/comments_anonymized.csv --output STEP3_Labeling/LLM_Labeling/output_labeled_comments_gemini.csv

    python STEP3_Labeling/LLM_Labeling/label_gemini.py --input STEP2_Anonymize/posts_anonymized.csv --output STEP3_Labeling/LLM_Labeling/output_labeled_posts_gemini.csv --sleep_seconds 4 --retry_sleep 50 --resume
    
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gemini-3.5-flash"
DEFAULT_TEXT_COLUMNS = ("content_anonymized", "content")
OUTPUT_COLUMNS = (
    "sentiment_label",
    "sentiment_score",
    "sentiment_raw_label",
    "sentiment_raw_scores",
    "status",
)
TEXT_OUTPUT_COLUMNS = (
    "sentiment_label",
    "sentiment_raw_label",
    "sentiment_raw_scores",
    "status",
)
STATUS_COLUMN = "status"

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

RAW_LABELS = ("LABEL_0", "LABEL_1", "LABEL_2")
SYSTEM_INSTRUCTION = """
You are a strict Vietnamese sentiment classifier.
Classify the user's anonymized Vietnamese students social-media text into exactly one label:
- negative
- neutral
- positive

Return a raw JSON object only. Do not add markdown, code fences, headings, or explanations.
The JSON must have:
- label: one of negative, neutral, positive, LABEL_0, LABEL_1, LABEL_2
- scores: an array with exactly 3 items for LABEL_0, LABEL_1, LABEL_2.

Label mapping:
LABEL_0 = negative
LABEL_1 = neutral
LABEL_2 = positive
"""

BATCH_SYSTEM_INSTRUCTION = """
You are a strict Vietnamese sentiment classifier.
Classify every anonymized Vietnamese students social-media text in the user's JSON input.

Return a raw JSON object only. Do not add markdown, code fences, headings, or explanations.
The JSON must have:
- items: an array with one result for every input item.

Each item must have:
- id: the same integer id from the input item.
- label: one of negative, neutral, positive, LABEL_0, LABEL_1, LABEL_2.
- scores: an array with exactly 3 items for LABEL_0, LABEL_1, LABEL_2.

Label mapping:
LABEL_0 = negative
LABEL_1 = neutral
LABEL_2 = positive
"""

LABEL_ONLY_SYSTEM_INSTRUCTION = """
You are a strict Vietnamese sentiment classifier.
Return exactly one token and nothing else:
LABEL_0 for negative
LABEL_1 for neutral
LABEL_2 for positive
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": ["label", "score"],
            },
        },
    },
    "required": ["label", "scores"],
}

BATCH_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "label": {"type": "string"},
                    "scores": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "score": {"type": "number"},
                            },
                            "required": ["label", "score"],
                        },
                    },
                },
                "required": ["id", "label", "scores"],
            },
        },
    },
    "required": ["items"],
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


def build_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


def parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Gemini did not return JSON: {text!r}")
    return json.loads(cleaned[start : end + 1])


def prediction_from_parsed(parsed: dict) -> dict:
    raw_label = normalize_raw_label(str(parsed.get("label", "")))
    if raw_label not in RAW_LABELS:
        raise ValueError(f"Unexpected Gemini label: {parsed.get('label')!r}")

    scores = normalize_scores(parsed.get("scores", []), raw_label)
    best_score = next(
        (item["score"] for item in scores if item["label"] == raw_label),
        scores[0]["score"],
    )

    return {
        "sentiment_label": normalize_label(raw_label),
        "sentiment_score": float(best_score),
        "sentiment_raw_label": raw_label,
        "sentiment_raw_scores": json.dumps(scores, ensure_ascii=False),
        STATUS_COLUMN: "done",
    }


def parse_label_response(text: str) -> str:
    cleaned = text.strip()
    raw_match = re.search(r"\bLABEL_[012]\b", cleaned.upper())
    if raw_match:
        return raw_match.group(0)

    normalized_match = re.search(
        r"\b(negative|neutral|positive|neg|neu|pos)\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if normalized_match:
        return normalize_raw_label(normalized_match.group(1))

    raise ValueError(f"Gemini did not return a valid label: {text!r}")


def normalize_scores(raw_scores: object, selected_raw_label: str) -> list[dict]:
    score_by_label = {label: 0.0 for label in RAW_LABELS}

    if isinstance(raw_scores, list):
        for item in raw_scores:
            if not isinstance(item, dict):
                continue
            raw_label = normalize_raw_label(str(item.get("label", "")))
            if raw_label not in score_by_label:
                continue
            try:
                score_by_label[raw_label] = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                score_by_label[raw_label] = 0.0

    if all(score <= 0.0 for score in score_by_label.values()):
        score_by_label[selected_raw_label] = 1.0

    scores = [
        {"label": label, "score": max(0.0, min(1.0, score_by_label[label]))}
        for label in RAW_LABELS
    ]
    return sorted(scores, key=lambda item: item["score"], reverse=True)


def call_gemini(
    client: genai.Client,
    text: str,
    model_name: str,
    max_retries: int,
    retry_sleep: float,
) -> dict:
    prompt = (
        "Classify sentiment for this Vietnamese text.\n"
        "Return exactly this JSON shape and start your answer with '{':\n"
        '{"label":"LABEL_1","scores":[{"label":"LABEL_0","score":0.0},'
        '{"label":"LABEL_1","score":1.0},{"label":"LABEL_2","score":0.0}]}\n\n'
        f"Text:\n{text}"
    )
    config = types.GenerateContentConfig(
        systemInstruction=SYSTEM_INSTRUCTION,
        temperature=0.0,
        maxOutputTokens=512,
        responseMimeType="application/json",
        responseSchema=RESPONSE_SCHEMA,
    )
    label_only_prompt = (
        "Classify sentiment for this Vietnamese text.\n"
        "Return exactly one token: LABEL_0, LABEL_1, or LABEL_2.\n\n"
        f"Text:\n{text}"
    )
    label_only_config = types.GenerateContentConfig(
        systemInstruction=LABEL_ONLY_SYSTEM_INSTRUCTION,
        temperature=0.0,
        maxOutputTokens=16,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            try:
                return parse_json_response(response.text or "")
            except ValueError as json_error:
                last_error = json_error

            fallback_response = client.models.generate_content(
                model=model_name,
                contents=label_only_prompt,
                config=label_only_config,
            )
            raw_label = parse_label_response(fallback_response.text or "")
            return {
                "label": raw_label,
                "scores": [
                    {"label": label, "score": 1.0 if label == raw_label else 0.0}
                    for label in RAW_LABELS
                ],
            }
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep * (attempt + 1))

    raise RuntimeError(f"Gemini labeling failed after {max_retries + 1} attempts") from last_error


def call_gemini_batch(
    client: genai.Client,
    items: list[dict],
    model_name: str,
    max_retries: int,
    retry_sleep: float,
) -> dict[int, dict]:
    if not items:
        return {}

    prompt = (
        "Classify sentiment for each item in this JSON array.\n"
        "Return exactly this JSON shape and include every input id once:\n"
        '{"items":[{"id":0,"label":"LABEL_1","scores":[{"label":"LABEL_0","score":0.0},'
        '{"label":"LABEL_1","score":1.0},{"label":"LABEL_2","score":0.0}]}]}\n\n'
        "Input JSON:\n"
        f"{json.dumps(items, ensure_ascii=False)}"
    )
    config = types.GenerateContentConfig(
        systemInstruction=BATCH_SYSTEM_INSTRUCTION,
        temperature=0.0,
        maxOutputTokens=max(1024, 512 * len(items)),
        responseMimeType="application/json",
        responseSchema=BATCH_RESPONSE_SCHEMA,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            parsed = parse_json_response(response.text or "")
            raw_items = parsed.get("items", [])
            if not isinstance(raw_items, list):
                raise ValueError(f"Gemini batch response has invalid items: {parsed!r}")

            result: dict[int, dict] = {}
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                try:
                    item_id = int(item.get("id"))
                except (TypeError, ValueError):
                    continue
                result[item_id] = item
            return result
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep * (attempt + 1))

    raise RuntimeError(f"Gemini batch labeling failed after {max_retries + 1} attempts") from last_error


def empty_prediction(status: str = "skipped") -> dict:
    return {
        "sentiment_label": "",
        "sentiment_score": 0.0,
        "sentiment_raw_label": "",
        "sentiment_raw_scores": "[]",
        STATUS_COLUMN: status,
    }


def predict_one(
    client: genai.Client,
    text: str,
    model_name: str,
    max_retries: int,
    retry_sleep: float,
) -> dict:
    if not text:
        return empty_prediction()

    parsed = call_gemini(
        client=client,
        text=text,
        model_name=model_name,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )
    return prediction_from_parsed(parsed)

def predict_batch(
    client: genai.Client,
    batch_items: list[tuple[int, str]],
    model_name: str,
    max_retries: int,
    retry_sleep: float,
) -> dict[int, dict]:
    request_items = [{"id": row_idx, "text": text} for row_idx, text in batch_items]
    parsed_by_id = call_gemini_batch(
        client=client,
        items=request_items,
        model_name=model_name,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )

    predictions: dict[int, dict] = {}
    for row_idx, text in batch_items:
        parsed = parsed_by_id.get(row_idx)
        if parsed is None:
            predictions[row_idx] = predict_one(
                client=client,
                text=text,
                model_name=model_name,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
            )
            continue
        predictions[row_idx] = prediction_from_parsed(parsed)
    return predictions


def load_resume_output(output_path: str | Path, expected_rows: int) -> pd.DataFrame | None:
    path = Path(output_path)
    if not path.exists():
        return None

    try:
        existing = read_table(path)
    except Exception:
        return None

    if len(existing) != expected_rows:
        return None
    return existing


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in TEXT_OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].astype("object")

    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0
        
    # ADD .astype(float) HERE to force the column dtype to float64
    df["sentiment_score"] = pd.to_numeric(
        df["sentiment_score"],
        errors="coerce",
    ).fillna(0.0).astype(float)

    raw_label_has_value = df["sentiment_raw_label"].fillna("").astype(str).str.strip() != ""
    status = df[STATUS_COLUMN].fillna("").astype(str).str.strip().str.lower()
    df[STATUS_COLUMN] = status
    df.loc[status == "", STATUS_COLUMN] = "pending"
    df.loc[raw_label_has_value & status.isin(["", "pending", "processing", "error"]), STATUS_COLUMN] = "done"
    df.loc[(df[STATUS_COLUMN] == "done") & ~raw_label_has_value, STATUS_COLUMN] = "pending"

    return df


def should_label_row(row: pd.Series, force_relabel: bool = False) -> bool:
    if force_relabel:
        return True
    status = str(row.get(STATUS_COLUMN, "")).strip().lower()
    return status not in {"done", "skipped"}


def apply_prediction(df: pd.DataFrame, row_idx: int, prediction: dict) -> None:
    for column, value in prediction.items():
        df.loc[row_idx, column] = value


def batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def label_dataframe(
    df: pd.DataFrame,
    text_col: str,
    output_path: str | Path,
    model_name: str = MODEL_NAME,
    min_len: int = 1,
    batch_size: int = 30,
    sleep_seconds: float = 0.0,
    max_retries: int = 3,
    retry_sleep: float = 5.0,
    save_every: int = 25,
    resume: bool = False,
    start: int = 0,
    limit: int | None = None,
    force_relabel: bool = False,
) -> pd.DataFrame:
    result_df = None if force_relabel else load_resume_output(output_path, len(df))
    if result_df is None:
        result_df = df.copy()
    result_df = ensure_output_columns(result_df)

    client = build_client()
    texts = [clean_text(value) for value in result_df[text_col].tolist()]

    end = len(result_df) if limit is None else min(len(result_df), start + limit)
    row_indexes = list(range(max(0, start), end))

    rows_to_label: list[tuple[int, str]] = []
    skipped_since_save = 0
    for row_idx in row_indexes:
        if not should_label_row(result_df.loc[row_idx], force_relabel=force_relabel):
            continue

        text = texts[row_idx]
        if len(text) < min_len:
            apply_prediction(result_df, row_idx, empty_prediction(status="skipped"))
            skipped_since_save += 1
            if save_every > 0 and skipped_since_save >= save_every:
                write_table(result_df, output_path)
                skipped_since_save = 0
            continue

        result_df.loc[row_idx, STATUS_COLUMN] = "pending"
        rows_to_label.append((row_idx, text))

    write_table(result_df, output_path)

    for batch in tqdm(
        list(batched(rows_to_label, max(1, batch_size))),
        desc="Labeling sentiment with Gemini",
        unit="batch",
    ):
        for row_idx, _ in batch:
            result_df.loc[row_idx, STATUS_COLUMN] = "processing"
        write_table(result_df, output_path)

        try:
            predictions = predict_batch(
                client=client,
                batch_items=batch,
                model_name=model_name,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
            )
        except Exception:
            for row_idx, _ in batch:
                result_df.loc[row_idx, STATUS_COLUMN] = "error"
            write_table(result_df, output_path)
            raise

        for row_idx, _ in batch:
            apply_prediction(result_df, row_idx, predictions[row_idx])

        write_table(result_df, output_path)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    write_table(result_df, output_path)
    return result_df


def run(
    input_path: str,
    output_path: str,
    text_col: str | None = None,
    model_name: str = MODEL_NAME,
    min_len: int = 1,
    batch_size: int = 30,
    sleep_seconds: float = 0.0,
    max_retries: int = 3,
    retry_sleep: float = 5.0,
    save_every: int = 25,
    resume: bool = False,
    start: int = 0,
    limit: int | None = None,
    force_relabel: bool = False,
) -> pd.DataFrame:
    df = read_table(input_path)
    selected_text_col = choose_text_column(df, text_col)

    return label_dataframe(
        df=df,
        text_col=selected_text_col,
        output_path=output_path,
        model_name=model_name,
        min_len=min_len,
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        save_every=save_every,
        resume=resume,
        start=start,
        limit=limit,
        force_relabel=force_relabel,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label Vietnamese sentiment with Gemini Flash."
    )
    parser.add_argument("--input", required=True, help="Input CSV/XLSX from STEP2_Anonymize")
    parser.add_argument("--output", required=True, help="Output CSV/XLSX with sentiment labels")
    parser.add_argument(
        "--text_col",
        default=None,
        help="Text column to label. Default: content_anonymized, then content",
    )
    parser.add_argument("--model", default=MODEL_NAME, help="Gemini model name")
    parser.add_argument("--min_len", type=int, default=1)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=30,
        help="Number of rows to send to Gemini in one request.",
    )
    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=0.0,
        help="Delay between Gemini calls to reduce quota pressure.",
    )
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=5.0)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Deprecated: resume is automatic when the output file already exists.",
    )
    parser.add_argument(
        "--force_relabel",
        action="store_true",
        help="Ignore existing status values and label the selected rows again.",
    )
    parser.add_argument("--start", type=int, default=0, help="Zero-based row index to start labeling.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to label.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        text_col=args.text_col,
        model_name=args.model,
        min_len=args.min_len,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        save_every=args.save_every,
        resume=args.resume,
        start=args.start,
        limit=args.limit,
        force_relabel=args.force_relabel,
    )
