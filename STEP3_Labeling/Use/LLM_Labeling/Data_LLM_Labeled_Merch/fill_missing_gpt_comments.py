# -*- coding: utf-8 -*-
"""
Fill missing GPT5.5_Medium labels in comments_dataset_labeled_merged.csv by id.

Example:
    python STEP3_Labeling/LLM_Labeling/Data_LLM_Labeled_Merch/fill_missing_gpt_comments.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET_PATH = SCRIPT_DIR / "comments_dataset_labeled_merged.csv"
DEFAULT_SOURCE_PATH = SCRIPT_DIR / "comments_dataset_labeled_merged_missingGPT_labeled.csv"
LABEL_COLUMN = "GPT5.5_Medium"
REQUIRED_COLUMNS = ["id", LABEL_COLUMN]


def is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


def fill_missing_gpt(
    target_path: str | Path = DEFAULT_TARGET_PATH,
    source_path: str | Path = DEFAULT_SOURCE_PATH,
) -> pd.DataFrame:
    target_path = Path(target_path)
    source_path = Path(source_path)

    target_df = pd.read_csv(target_path)
    source_df = pd.read_csv(source_path)

    for path, df in ((target_path, target_df), (source_path, source_df)):
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing columns in {path}: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )

    duplicated_count = int(source_df["id"].duplicated().sum())
    if duplicated_count:
        raise ValueError(f"{source_path} has {duplicated_count} duplicated id values")

    source_labels = source_df.set_index("id")[LABEL_COLUMN]
    target_ids = target_df["id"]
    source_values = target_ids.map(source_labels)

    fill_mask = is_blank(target_df[LABEL_COLUMN]) & ~is_blank(source_values)
    target_df.loc[fill_mask, LABEL_COLUMN] = source_values[fill_mask]

    target_df.to_csv(target_path, index=False, encoding="utf-8-sig")

    print(f"Source rows              : {len(source_df)}")
    print(f"Target rows              : {len(target_df)}")
    print(f"Filled GPT5.5_Medium rows: {int(fill_mask.sum())}")
    print(f"Remaining missing GPT    : {int(is_blank(target_df[LABEL_COLUMN]).sum())}")
    return target_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill missing GPT5.5_Medium labels in merged comments by id."
    )
    parser.add_argument("--target", default=str(DEFAULT_TARGET_PATH))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fill_missing_gpt(
        target_path=args.target,
        source_path=args.source,
    )
