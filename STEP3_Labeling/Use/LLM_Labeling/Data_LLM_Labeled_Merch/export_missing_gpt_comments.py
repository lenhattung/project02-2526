# -*- coding: utf-8 -*-
"""
Export comment rows that are missing GPT5.5_Medium labels.

Output columns:
    id, content_anonymized, GPT5.5_Medium

Example:
    python STEP3_Labeling/LLM_Labeling/Data_LLM_Labeled_Merch/export_missing_gpt_comments.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIR / "comments_dataset_labeled_merged.csv"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "comments_dataset_labeled_merged_missingGPT.csv"
OUTPUT_COLUMNS = ["id", "content_anonymized", "GPT5.5_Medium"]


def export_missing_gpt(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing columns in {input_path}: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    has_id = df["id"].notna() & (df["id"].astype(str).str.strip() != "")
    missing_gpt = df["GPT5.5_Medium"].isna() | (
        df["GPT5.5_Medium"].astype(str).str.strip() == ""
    )
    result = df.loc[has_id & missing_gpt, OUTPUT_COLUMNS].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export comments with missing GPT5.5_Medium labels."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_df = export_missing_gpt(
        input_path=args.input,
        output_path=args.output,
    )
    print(f"Saved {len(result_df)} rows to {args.output}")
