# -*- coding: utf-8 -*-
"""
Compare sentiment labels from two labeled CSV/XLSX files with Cohen's Kappa.

Example:
    python STEP3_Labeling/kappa_compare.py

    python STEP3_Labeling/kappa_compare.py ^
        --file_a STEP3_Labeling/output_labeled_comments_vsfc_phobert.csv ^
        --file_b STEP3_Labeling/output_labeled_comments_visobert.csv ^
        --label_col sentiment_label ^
        --key_col id


python STEP3_Labeling/Kappa_compare.py --output STEP3_Labeling/kappa_report_posts.json




"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FILE_A = SCRIPT_DIR / "output_labeled_posts_vsfc_phobert.csv"
DEFAULT_FILE_B = SCRIPT_DIR / "output_labeled_posts_visobert.csv"
DEFAULT_LABEL_COL = "sentiment_raw_label"
AUTO_KEY_COLUMNS = ("id", "comment_id", "post_id")


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
    return LABEL_NORMALIZATION.get(label, LABEL_NORMALIZATION.get(label.upper(), label))


def validate_label_column(df: pd.DataFrame, label_col: str, file_label: str) -> None:
    if label_col not in df.columns:
        raise ValueError(
            f"Column '{label_col}' not found in {file_label}. "
            f"Available columns: {list(df.columns)}"
        )


def choose_key_column(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    requested_key_col: str | None,
) -> str | None:
    if requested_key_col:
        missing = [
            name
            for name, df in (("file_a", df_a), ("file_b", df_b))
            if requested_key_col not in df.columns
        ]
        if missing:
            raise ValueError(
                f"Key column '{requested_key_col}' not found in: {', '.join(missing)}"
            )
        return requested_key_col

    for column in AUTO_KEY_COLUMNS:
        if column in df_a.columns and column in df_b.columns:
            return column

    return None


def prepare_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_col_a: str,
    label_col_b: str,
    key_col: str | None,
    drop_empty: bool,
) -> pd.DataFrame:
    if key_col:
        left = df_a[[key_col, label_col_a]].rename(columns={label_col_a: "label_a"})
        right = df_b[[key_col, label_col_b]].rename(columns={label_col_b: "label_b"})

        if left[key_col].duplicated().any():
            raise ValueError(f"Duplicate key values found in file_a column '{key_col}'")
        if right[key_col].duplicated().any():
            raise ValueError(f"Duplicate key values found in file_b column '{key_col}'")

        compared = left.merge(right, on=key_col, how="inner")
    else:
        if len(df_a) != len(df_b):
            raise ValueError(
                "Files have different row counts and no shared key column was found. "
                "Pass --key_col, or make sure both files have the same row count."
            )
        compared = pd.DataFrame(
            {
                "row_index": range(len(df_a)),
                "label_a": df_a[label_col_a].to_numpy(),
                "label_b": df_b[label_col_b].to_numpy(),
            }
        )

    compared["label_a"] = compared["label_a"].map(normalize_label)
    compared["label_b"] = compared["label_b"].map(normalize_label)

    if drop_empty:
        compared = compared[
            (compared["label_a"] != "") & (compared["label_b"] != "")
        ].copy()

    return compared.reset_index(drop=True)


def cohen_kappa(labels_a: pd.Series, labels_b: pd.Series) -> tuple[float, float, float]:
    total = len(labels_a)
    if total == 0:
        raise ValueError("No comparable rows after filtering labels.")

    observed_agreement = (labels_a == labels_b).sum() / total
    counts_a = labels_a.value_counts(normalize=True)
    counts_b = labels_b.value_counts(normalize=True)
    all_labels = sorted(set(counts_a.index).union(counts_b.index))
    expected_agreement = sum(counts_a.get(label, 0.0) * counts_b.get(label, 0.0) for label in all_labels)

    if expected_agreement == 1.0:
        kappa = 1.0 if observed_agreement == 1.0 else 0.0
    else:
        kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)

    return kappa, observed_agreement, expected_agreement


def build_report(compared: pd.DataFrame) -> dict:
    kappa, observed_agreement, expected_agreement = cohen_kappa(
        compared["label_a"],
        compared["label_b"],
    )
    confusion = pd.crosstab(
        compared["label_a"],
        compared["label_b"],
        rownames=["file_a"],
        colnames=["file_b"],
        dropna=False,
    )

    return {
        "total_compared": int(len(compared)),
        "agree_count": int((compared["label_a"] == compared["label_b"]).sum()),
        "disagree_count": int((compared["label_a"] != compared["label_b"]).sum()),
        "observed_agreement": observed_agreement,
        "expected_agreement": expected_agreement,
        "cohen_kappa": kappa,
        "file_a_label_counts": compared["label_a"].value_counts().sort_index().to_dict(),
        "file_b_label_counts": compared["label_b"].value_counts().sort_index().to_dict(),
        "confusion_matrix": confusion,
    }


def print_report(report: dict, key_col: str | None) -> None:
    print("=== Cohen's Kappa comparison ===")
    print(f"Alignment       : {key_col if key_col else 'row order'}")
    print(f"Rows compared   : {report['total_compared']}")
    print(f"Agree / Disagree: {report['agree_count']} / {report['disagree_count']}")
    print(f"Observed agree  : {report['observed_agreement']:.4f}")
    print(f"Expected agree  : {report['expected_agreement']:.4f}")
    print(f"Cohen's Kappa   : {report['cohen_kappa']:.4f}")
    print()
    print("Label counts - file_a:")
    print(pd.Series(report["file_a_label_counts"]).to_string())
    print()
    print("Label counts - file_b:")
    print(pd.Series(report["file_b_label_counts"]).to_string())
    print()
    print("Confusion matrix:")
    print(report["confusion_matrix"].to_string())


def save_report(report: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        report["confusion_matrix"].to_csv(path, encoding="utf-8-sig")
        return

    serializable = {
        key: value
        for key, value in report.items()
        if key != "confusion_matrix"
    }
    serializable["confusion_matrix"] = report["confusion_matrix"].to_dict()
    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure agreement between two labeled sentiment files with Cohen's Kappa."
    )
    parser.add_argument("--file_a", default=str(DEFAULT_FILE_A), help="First labeled CSV/XLSX")
    parser.add_argument("--file_b", default=str(DEFAULT_FILE_B), help="Second labeled CSV/XLSX")
    parser.add_argument(
        "--label_col",
        default=DEFAULT_LABEL_COL,
        help="Label column name used by both files. Default: sentiment_label",
    )
    parser.add_argument(
        "--label_col_a",
        default=None,
        help="Label column in file_a if different from --label_col",
    )
    parser.add_argument(
        "--label_col_b",
        default=None,
        help="Label column in file_b if different from --label_col",
    )
    parser.add_argument(
        "--key_col",
        default=None,
        help="Column used to align rows. Default: auto-detect id/comment_id/post_id, else row order.",
    )
    parser.add_argument(
        "--keep_empty",
        action="store_true",
        help="Keep rows with empty labels. Default: drop rows where either label is empty.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional report path. Use .csv for confusion matrix only, otherwise JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_col_a = args.label_col_a or args.label_col
    label_col_b = args.label_col_b or args.label_col

    df_a = read_table(args.file_a)
    df_b = read_table(args.file_b)
    validate_label_column(df_a, label_col_a, "file_a")
    validate_label_column(df_b, label_col_b, "file_b")

    key_col = choose_key_column(df_a, df_b, args.key_col)
    compared = prepare_comparison(
        df_a=df_a,
        df_b=df_b,
        label_col_a=label_col_a,
        label_col_b=label_col_b,
        key_col=key_col,
        drop_empty=not args.keep_empty,
    )
    report = build_report(compared)
    print_report(report, key_col)

    if args.output:
        save_report(report, args.output)
        print()
        print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
