# -*- coding: utf-8 -*-
"""
Create final merchant vote file from pairwise merged labels.

Default usage:
    python "STEP3_Labeling/pairwise merch/export_vote_final_merch.py"

Output:
    STEP3_Labeling/pairwise merch/output_vote_final_merch.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_PATH = SCRIPT_DIR / "kappa_pairwise_merch_merged_by_id.csv"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "output_vote_final_merch.csv"

MODEL_A_COLUMN = "3.5 Flash"
MODEL_B_COLUMN = "3.1 Pro"
MODEL_C_COLUMN = "GPT5.5_Medium"

OUTPUT_COLUMNS = [
    "data_types",
    "record_key",
    "id",
    "source",
    "posted_at",
    "content",
    "content_anonymized",
    MODEL_A_COLUMN,
    MODEL_B_COLUMN,
    MODEL_C_COLUMN,
    "is_Human",
    "label",
]


def read_table(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    suffix = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {suffix}. Use .csv, .xlsx, or .xls")


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    label = " ".join(str(value).strip().split())
    if label.endswith(".0") and label[:-2].isdigit():
        return label[:-2]
    return label


def validate_input(df: pd.DataFrame) -> None:
    required_columns = [
        "data_type",
        "record_key",
        "id",
        "source",
        "posted_at",
        "content",
        "content_anonymized",
        MODEL_A_COLUMN,
        MODEL_B_COLUMN,
        MODEL_C_COLUMN,
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def build_vote_final(df: pd.DataFrame) -> pd.DataFrame:
    validate_input(df)

    result = pd.DataFrame()
    result["data_types"] = df["data_type"]
    for column in [
        "record_key",
        "id",
        "source",
        "posted_at",
        "content",
        "content_anonymized",
    ]:
        result[column] = df[column]

    result[MODEL_A_COLUMN] = df[MODEL_A_COLUMN].map(normalize_label)
    result[MODEL_B_COLUMN] = df[MODEL_B_COLUMN].map(normalize_label)
    result[MODEL_C_COLUMN] = df[MODEL_C_COLUMN].map(normalize_label)

    agreed = (
        (result[MODEL_A_COLUMN] != "")
        & (result[MODEL_B_COLUMN] != "")
        & (result[MODEL_A_COLUMN] == result[MODEL_B_COLUMN])
    )
    result["is_Human"] = ~agreed
    result["label"] = ""
    result.loc[agreed, "label"] = result.loc[agreed, MODEL_A_COLUMN]

    return result[OUTPUT_COLUMNS]


def run(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    df = read_table(input_path)
    result = build_vote_final(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export final merchant vote labels from 3.5 Flash and 3.1 Pro agreement."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(input_path=args.input, output_path=args.output)

    needs_human = int(result["is_Human"].sum())
    auto_labeled = int((~result["is_Human"]).sum())
    print("=== Vote final export complete ===")
    print(f"Output      : {args.output}")
    print(f"Total rows  : {len(result)}")
    print(f"Auto labels : {auto_labeled}")
    print(f"Need human  : {needs_human}")


if __name__ == "__main__":
    main()
