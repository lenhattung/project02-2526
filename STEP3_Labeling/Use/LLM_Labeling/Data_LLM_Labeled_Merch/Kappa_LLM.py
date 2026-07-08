# -*- coding: utf-8 -*-
"""
Compute pairwise Cohen's Kappa for LLM labels and build majority-vote output.

Default usage:
    python STEP3_Labeling/LLM_Labeling/Data_LLM_Labeled_Merch/Kappa_LLM.py

Inputs:
    - comments_dataset_labeled_merged.csv
    - posts_dataset_labeled_merged.csv

Outputs:
    - llm_kappa_merged_all.csv
    - llm_kappa_label_counts.csv
    - llm_kappa_pairwise.csv
    - llm_kappa_report.json
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_COMMENTS_PATH = SCRIPT_DIR / "comments_dataset_labeled_merged.csv"
DEFAULT_POSTS_PATH = SCRIPT_DIR / "posts_dataset_labeled_merged.csv"

DEFAULT_OUTPUT_MERGED = SCRIPT_DIR / "llm_kappa_merged_all.csv"
DEFAULT_OUTPUT_COUNTS = SCRIPT_DIR / "llm_kappa_label_counts.csv"
DEFAULT_OUTPUT_PAIRWISE = SCRIPT_DIR / "llm_kappa_pairwise.csv"
DEFAULT_OUTPUT_REPORT = SCRIPT_DIR / "llm_kappa_report.json"

ID_COLUMN = "id"
TEXT_COLUMN = "content_anonymized"
MODEL_COLUMNS = ("3.5 Flash", "3.1 Pro", "GPT5.5_Medium")
FINAL_COLUMNS = [
    ID_COLUMN,
    TEXT_COLUMN,
    "3.5 Flash",
    "3.1 Pro",
    "GPT5.5_Medium",
    "label",
]

LABEL_NORMALIZATION = {
    "0": "0",
    "0.0": "0",
    "LABEL_0": "0",
    "NEG": "0",
    "NEGATIVE": "0",
    "negative": "0",
    "1": "1",
    "1.0": "1",
    "LABEL_1": "1",
    "NEU": "1",
    "NEUTRAL": "1",
    "neutral": "1",
    "2": "2",
    "2.0": "2",
    "LABEL_2": "2",
    "POS": "2",
    "POSITIVE": "2",
    "positive": "2",
}


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
    if not label:
        return ""
    return LABEL_NORMALIZATION.get(label, LABEL_NORMALIZATION.get(label.upper(), label))


def validate_input(df: pd.DataFrame, file_label: str) -> None:
    required_columns = [ID_COLUMN, TEXT_COLUMN, *MODEL_COLUMNS]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{file_label} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def prepare_dataset(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    result = df[[ID_COLUMN, TEXT_COLUMN, *MODEL_COLUMNS]].copy()
    for column in MODEL_COLUMNS:
        result[column] = result[column].map(normalize_label)
    result["label"] = result.apply(majority_label, axis=1)
    result["data_type"] = data_type
    return result


def majority_label(row: pd.Series) -> str:
    labels = [str(row[column]).strip() for column in MODEL_COLUMNS]
    labels = [label for label in labels if label]
    if len(labels) < 2:
        return ""

    counts = pd.Series(labels).value_counts()
    top_count = int(counts.iloc[0])
    if top_count >= 2:
        return str(counts.index[0])
    return ""


def cohen_kappa(labels_a: pd.Series, labels_b: pd.Series) -> tuple[float, float, float]:
    total = len(labels_a)
    if total == 0:
        return 0.0, 0.0, 0.0

    observed_agreement = float((labels_a == labels_b).sum() / total)
    counts_a = labels_a.value_counts(normalize=True)
    counts_b = labels_b.value_counts(normalize=True)
    all_labels = sorted(set(counts_a.index).union(counts_b.index))
    expected_agreement = float(
        sum(counts_a.get(label, 0.0) * counts_b.get(label, 0.0) for label in all_labels)
    )

    if expected_agreement == 1.0:
        kappa = 1.0 if observed_agreement == 1.0 else 0.0
    else:
        kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)

    return float(kappa), observed_agreement, expected_agreement


def compare_pair(group: pd.DataFrame, data_type: str, model_a: str, model_b: str) -> dict:
    comparable = group[
        (group[model_a].fillna("").astype(str).str.strip() != "")
        & (group[model_b].fillna("").astype(str).str.strip() != "")
    ].copy()

    kappa, observed, expected = cohen_kappa(comparable[model_a], comparable[model_b])
    agree_count = int((comparable[model_a] == comparable[model_b]).sum())
    disagree_count = int(len(comparable) - agree_count)
    confusion = pd.crosstab(
        comparable[model_a],
        comparable[model_b],
        rownames=[model_a],
        colnames=[model_b],
        dropna=False,
    )

    return {
        "data_type": data_type,
        "model_a": model_a,
        "model_b": model_b,
        "rows": int(len(group)),
        "comparable_rows": int(len(comparable)),
        "agree_count": agree_count,
        "disagree_count": disagree_count,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "cohen_kappa": kappa,
        "confusion_matrix": confusion.to_dict(),
    }


def build_pairwise_report(merged_with_type: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pair_rows: list[dict] = []
    report: dict[str, object] = {
        "total_rows": int(len(merged_with_type)),
        "by_data_type": {},
    }

    for data_type, group in merged_with_type.groupby("data_type", dropna=False):
        data_type_key = str(data_type)
        report["by_data_type"][data_type_key] = {}

        for model_a, model_b in combinations(MODEL_COLUMNS, 2):
            item = compare_pair(group, data_type_key, model_a, model_b)
            report["by_data_type"][data_type_key][f"{model_a} vs {model_b}"] = item
            pair_rows.append(
                {
                    key: value
                    for key, value in item.items()
                    if key != "confusion_matrix"
                }
            )

    report["overall"] = {}
    for model_a, model_b in combinations(MODEL_COLUMNS, 2):
        item = compare_pair(merged_with_type, "overall", model_a, model_b)
        report["overall"][f"{model_a} vs {model_b}"] = item
        pair_rows.append(
            {
                key: value
                for key, value in item.items()
                if key != "confusion_matrix"
            }
        )

    return pd.DataFrame(pair_rows), report


def build_label_counts(merged_with_type: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for data_type, group in merged_with_type.groupby("data_type", dropna=False):
        for model_name in (*MODEL_COLUMNS, "label"):
            counts = group[model_name].dropna().astype(str).str.strip()
            counts = counts[counts != ""].value_counts().sort_index()
            for label, count in counts.items():
                rows.append(
                    {
                        "data_type": str(data_type),
                        "model": model_name,
                        "label": label,
                        "count": int(count),
                    }
                )
    return pd.DataFrame(rows)


def write_outputs(
    merged: pd.DataFrame,
    counts: pd.DataFrame,
    pairwise: pd.DataFrame,
    report: dict,
    output_merged: str | Path,
    output_counts: str | Path,
    output_pairwise: str | Path,
    output_report: str | Path,
) -> None:
    output_merged = Path(output_merged)
    output_counts = Path(output_counts)
    output_pairwise = Path(output_pairwise)
    output_report = Path(output_report)

    output_merged.parent.mkdir(parents=True, exist_ok=True)
    output_counts.parent.mkdir(parents=True, exist_ok=True)
    output_pairwise.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(output_merged, index=False, encoding="utf-8-sig")
    counts.to_csv(output_counts, index=False, encoding="utf-8-sig")
    pairwise.to_csv(output_pairwise, index=False, encoding="utf-8-sig")
    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(
    comments_path: str | Path = DEFAULT_COMMENTS_PATH,
    posts_path: str | Path = DEFAULT_POSTS_PATH,
    output_merged: str | Path = DEFAULT_OUTPUT_MERGED,
    output_counts: str | Path = DEFAULT_OUTPUT_COUNTS,
    output_pairwise: str | Path = DEFAULT_OUTPUT_PAIRWISE,
    output_report: str | Path = DEFAULT_OUTPUT_REPORT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    comments_df = read_table(comments_path)
    posts_df = read_table(posts_path)
    validate_input(comments_df, "comments")
    validate_input(posts_df, "posts")

    comments = prepare_dataset(comments_df, "comment")
    posts = prepare_dataset(posts_df, "post")
    merged_with_type = pd.concat([posts, comments], ignore_index=True)

    pairwise, report = build_pairwise_report(merged_with_type)
    counts = build_label_counts(merged_with_type)
    merged = merged_with_type[FINAL_COLUMNS].copy()

    write_outputs(
        merged=merged,
        counts=counts,
        pairwise=pairwise,
        report=report,
        output_merged=output_merged,
        output_counts=output_counts,
        output_pairwise=output_pairwise,
        output_report=output_report,
    )
    return merged, counts, pairwise, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pairwise Cohen's Kappa for 3 LLM label columns."
    )
    parser.add_argument("--comments", default=str(DEFAULT_COMMENTS_PATH))
    parser.add_argument("--posts", default=str(DEFAULT_POSTS_PATH))
    parser.add_argument("--output_merged", default=str(DEFAULT_OUTPUT_MERGED))
    parser.add_argument("--output_counts", default=str(DEFAULT_OUTPUT_COUNTS))
    parser.add_argument("--output_pairwise", default=str(DEFAULT_OUTPUT_PAIRWISE))
    parser.add_argument("--output_report", default=str(DEFAULT_OUTPUT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged, counts, pairwise, report = run(
        comments_path=args.comments,
        posts_path=args.posts,
        output_merged=args.output_merged,
        output_counts=args.output_counts,
        output_pairwise=args.output_pairwise,
        output_report=args.output_report,
    )

    print("=== LLM Kappa complete ===")
    print(f"Merged data : {args.output_merged}")
    print(f"Label counts: {args.output_counts}")
    print(f"Pairwise    : {args.output_pairwise}")
    print(f"Report      : {args.output_report}")
    print()
    print(f"Total rows  : {len(merged)}")
    print(f"Rows with majority label: {int((merged['label'].fillna('').astype(str).str.strip() != '').sum())}")
    print()
    for _, row in pairwise[pairwise["data_type"] == "overall"].iterrows():
        print(
            f"overall {row['model_a']} vs {row['model_b']}: "
            f"kappa={row['cohen_kappa']:.4f}, "
            f"agreement={row['observed_agreement']:.4f}, "
            f"n={int(row['comparable_rows'])}"
        )


if __name__ == "__main__":
    main()
