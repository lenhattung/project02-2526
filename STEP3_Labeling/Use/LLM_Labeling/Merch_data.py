# -*- coding: utf-8 -*-
"""
Merge post label files by id.

Input files:
    - Data_labeled/posts_anonymized_labeled_final _GPT5.5_Medium.csv
    - Data_labeled/post_dataset_labeled_gemini.xlsx

Output columns:
    id, content_anonymized, 3.5 Flash, 3.1 Pro, GPT5.5_Medium

Example:
    python STEP3_Labeling/LLM_Labeling/Merch_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Data_labeled"

DEFAULT_GPT_PATH = DATA_DIR / "posts_anonymized_labeled_final _GPT5.5_Medium.csv"
DEFAULT_GEMINI_PATH = DATA_DIR / "post_dataset_labeled_gemini.xlsx"
DEFAULT_OUTPUT_PATH = DATA_DIR / "post_dataset_labeled_merged.csv"

FINAL_COLUMNS = [
    "id",
    "content_anonymized",
    "3.5 Flash",
    "3.1 Pro",
    "GPT5.5_Medium",
]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {suffix}. Use .csv, .xlsx, or .xls")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
        return

    raise ValueError(f"Unsupported output file type: {suffix}. Use .csv, .xlsx, or .xls")


def require_columns(df: pd.DataFrame, columns: list[str], path: str | Path) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing columns in {path}: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )


def require_unique_id(df: pd.DataFrame, path: str | Path) -> None:
    duplicated_count = int(df["id"].duplicated().sum())
    if duplicated_count:
        raise ValueError(f"{path} has {duplicated_count} duplicated id values")


def merge_post_labels(
    gpt_path: str | Path = DEFAULT_GPT_PATH,
    gemini_path: str | Path = DEFAULT_GEMINI_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    gpt_df = read_table(gpt_path)
    gemini_df = read_table(gemini_path)

    require_columns(gpt_df, ["id", "content_anonymized", "GPT5.5_Medium"], gpt_path)
    require_columns(gemini_df, ["id", "content_anonymized", "3.5 Flash", "3.1 Pro"], gemini_path)
    require_unique_id(gpt_df, gpt_path)
    require_unique_id(gemini_df, gemini_path)

    merged = gemini_df[["id", "content_anonymized", "3.5 Flash", "3.1 Pro"]].merge(
        gpt_df[["id", "content_anonymized", "GPT5.5_Medium"]],
        on="id",
        how="outer",
        suffixes=("_gemini", "_gpt"),
        validate="one_to_one",
        indicator=True,
    )

    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        counts = unmatched["_merge"].value_counts().to_dict()
        raise ValueError(f"ID mismatch between files: {counts}")

    content_diff = (
        merged["content_anonymized_gemini"].fillna("").astype(str)
        != merged["content_anonymized_gpt"].fillna("").astype(str)
    )
    if content_diff.any():
        print(f"Warning: {int(content_diff.sum())} rows have different content_anonymized values.")

    merged["content_anonymized"] = merged["content_anonymized_gemini"].combine_first(
        merged["content_anonymized_gpt"]
    )
    result = merged[FINAL_COLUMNS].sort_values("id").reset_index(drop=True)
    write_table(result, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge GPT and Gemini post labels by id.")
    parser.add_argument("--gpt", default=str(DEFAULT_GPT_PATH), help="CSV with GPT5.5_Medium labels")
    parser.add_argument("--gemini", default=str(DEFAULT_GEMINI_PATH), help="XLSX/CSV with Gemini labels")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Merged output path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merged_df = merge_post_labels(
        gpt_path=args.gpt,
        gemini_path=args.gemini,
        output_path=args.output,
    )
    print(f"Saved {len(merged_df)} rows to {args.output}")
