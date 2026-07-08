# -*- coding: utf-8 -*-
"""
Split comments_anonymized.csv into 1K-row labeling files.

Each output file contains only:
    - id
    - content_anonymized
    - label

Example:
    python STEP3_Labeling/LLM_Labeling/split_data_comments1k.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "STEP2_Anonymize" / "comments_anonymized.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "split_file1K"
OUTPUT_COLUMNS = ["id", "content_anonymized", "label"]


def split_comments(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    chunk_size: int = 1000,
) -> list[Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    df = pd.read_csv(input_path)
    required_columns = ["id", "content_anonymized"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing columns in {input_path}: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    split_df = df[required_columns].copy()
    split_df["label"] = ""

    output_paths: list[Path] = []
    total_rows = len(split_df)
    for start in range(0, total_rows, chunk_size):
        part_number = (start // chunk_size) + 1
        chunk = split_df.iloc[start : start + chunk_size][OUTPUT_COLUMNS]
        output_path = output_dir / f"comments_anonymized_{part_number:03d}.csv"
        chunk.to_csv(output_path, index=False, encoding="utf-8-sig")
        output_paths.append(output_path)

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split comments_anonymized.csv into 1K-row CSV files for labeling."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input comments CSV")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output folder")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Rows per output file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = split_comments(
        input_path=args.input,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
    )
    print(f"Created {len(paths)} files in {Path(args.output_dir)}")
    for path in paths:
        print(path)
