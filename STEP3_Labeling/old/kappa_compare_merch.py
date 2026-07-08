# -*- coding: utf-8 -*-
"""
Merge post/comment sentiment outputs from phobert_student_vn and VSFC PhoBERT.

Default usage:
    python STEP3_Labeling/kappa_compare_merch.py

Outputs:
    - kappa_merged_all.csv: combined post + comment rows with both model labels
    - kappa_merged_label_counts.csv: label counts by data type and model
    - kappa_merged_report.json: agreement/kappa summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_COMMENTS_phobert_student_vn = SCRIPT_DIR / "output_labeled_comments_phobert_student_vn.csv"
DEFAULT_POSTS_phobert_student_vn = SCRIPT_DIR / "output_labeled_posts_phobert_student_vn.csv"
DEFAULT_POSTS_VSFC_PHOBERT = SCRIPT_DIR / "output_labeled_posts_vsfc_phobert.csv"
DEFAULT_COMMENTS_VSFC_PHOBERT = SCRIPT_DIR / "output_labeled_comments_vsfc_phobert.csv"

DEFAULT_OUTPUT_MERGED = SCRIPT_DIR / "kappa_merged_all.csv"
DEFAULT_OUTPUT_COUNTS = SCRIPT_DIR / "kappa_merged_label_counts.csv"
DEFAULT_OUTPUT_REPORT = SCRIPT_DIR / "kappa_merged_report.json"

LABEL_COL = "sentiment_label"
SCORE_COL = "sentiment_score"
RAW_LABEL_COL = "sentiment_raw_label"
RAW_SCORES_COL = "sentiment_raw_scores"

BASE_METADATA_COLUMNS = (
    "post_id",
    "content",
    "source",
    "posted_at",
    "like_count",
    "comment_count",
    "collected_at",
)

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


def choose_merge_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    if "id" not in df_a.columns or "id" not in df_b.columns:
        raise ValueError("Both files must have an 'id' column to merge safely.")

    if "content_anonymized" in df_a.columns and "content_anonymized" in df_b.columns:
        return ["id", "content_anonymized"]

    return ["id"]


def validate_input(df: pd.DataFrame, file_label: str) -> None:
    missing = [column for column in ("id", LABEL_COL) if column not in df.columns]
    if missing:
        raise ValueError(
            f"{file_label} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def prepare_model_side(
    df: pd.DataFrame,
    model_name: str,
    merge_columns: list[str],
    metadata_columns: list[str],
) -> pd.DataFrame:
    side = df[merge_columns].copy()

    for column in metadata_columns:
        if column in df.columns:
            side[f"{column}__{model_name}"] = df[column]

    side[f"{model_name}_label"] = df[LABEL_COL].map(normalize_label)
    side[f"{model_name}_score"] = df[SCORE_COL] if SCORE_COL in df.columns else pd.NA
    side[f"{model_name}_raw_label"] = (
        df[RAW_LABEL_COL] if RAW_LABEL_COL in df.columns else pd.NA
    )
    side[f"{model_name}_raw_scores"] = (
        df[RAW_SCORES_COL] if RAW_SCORES_COL in df.columns else pd.NA
    )
    return side


def first_available(merged: pd.DataFrame, column_names: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=merged.index)
    for column in column_names:
        if column in merged.columns:
            result = result.combine_first(merged[column])
    return result


def merge_dataset(
    data_type: str,
    phobert_student_vn_path: str | Path,
    vsfc_phobert_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    df_phobert_student_vn = read_table(phobert_student_vn_path)
    df_vsfc = read_table(vsfc_phobert_path)
    validate_input(df_phobert_student_vn, f"{data_type} phobert_student_vn")
    validate_input(df_vsfc, f"{data_type} VSFC PhoBERT")

    merge_columns = choose_merge_columns(df_phobert_student_vn, df_vsfc)
    metadata_columns = [
        column
        for column in BASE_METADATA_COLUMNS
        if column not in merge_columns
        and (column in df_phobert_student_vn.columns or column in df_vsfc.columns)
    ]

    phobert_student_vn_side = prepare_model_side(
        df_phobert_student_vn,
        "phobert_student_vn",
        merge_columns,
        metadata_columns,
    )
    vsfc_side = prepare_model_side(
        df_vsfc,
        "vsfc_phobert",
        merge_columns,
        metadata_columns,
    )

    merged = phobert_student_vn_side.merge(
        vsfc_side,
        on=merge_columns,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    merged.insert(0, "data_type", data_type)

    for column in metadata_columns:
        merged[column] = first_available(
            merged,
            [f"{column}__phobert_student_vn", f"{column}__vsfc_phobert"],
        )

    has_phobert_student_vn = merged["phobert_student_vn_label"].fillna("") != ""
    has_vsfc = merged["vsfc_phobert_label"].fillna("") != ""
    merged["merge_status"] = "both_models"
    merged.loc[has_phobert_student_vn & ~has_vsfc, "merge_status"] = "only_phobert_student_vn"
    merged.loc[~has_phobert_student_vn & has_vsfc, "merge_status"] = "only_vsfc_phobert"
    merged.loc[~has_phobert_student_vn & ~has_vsfc, "merge_status"] = "no_label"

    merged["models_agree"] = pd.NA
    comparable = has_phobert_student_vn & has_vsfc
    merged.loc[comparable, "models_agree"] = (
        merged.loc[comparable, "phobert_student_vn_label"]
        == merged.loc[comparable, "vsfc_phobert_label"]
    )

    merged["record_key"] = data_type + ":" + merged["id"].astype(str)

    drop_columns = [
        column
        for column in merged.columns
        if column.endswith("__phobert_student_vn") or column.endswith("__vsfc_phobert")
    ]
    merged = merged.drop(columns=drop_columns + ["_merge"])

    ordered_columns = [
        "data_type",
        "record_key",
        "id",
        "post_id",
        "content",
        "content_anonymized",
        "source",
        "posted_at",
        "like_count",
        "comment_count",
        "collected_at",
        "vsfc_phobert_label",
        "vsfc_phobert_score",
        "vsfc_phobert_raw_label",
        "phobert_student_vn_label",
        "phobert_student_vn_score",
        "phobert_student_vn_raw_label",
        "models_agree",
        "merge_status",
        "vsfc_phobert_raw_scores",
        "phobert_student_vn_raw_scores",
    ]
    ordered_columns = [column for column in ordered_columns if column in merged.columns]
    remaining_columns = [column for column in merged.columns if column not in ordered_columns]
    merged = merged[ordered_columns + remaining_columns]

    diagnostics = {
        "data_type": data_type,
        "phobert_student_vn_rows": int(len(df_phobert_student_vn)),
        "vsfc_phobert_rows": int(len(df_vsfc)),
        "merged_rows": int(len(merged)),
        "both_models_rows": int((merged["merge_status"] == "both_models").sum()),
        "only_phobert_student_vn_rows": int((merged["merge_status"] == "only_phobert_student_vn").sum()),
        "only_vsfc_phobert_rows": int(
            (merged["merge_status"] == "only_vsfc_phobert").sum()
        ),
        "merge_columns": merge_columns,
    }
    return merged, diagnostics


def cohen_kappa(labels_a: pd.Series, labels_b: pd.Series) -> tuple[float, float, float]:
    total = len(labels_a)
    if total == 0:
        return 0.0, 0.0, 0.0

    observed_agreement = (labels_a == labels_b).sum() / total
    counts_a = labels_a.value_counts(normalize=True)
    counts_b = labels_b.value_counts(normalize=True)
    all_labels = sorted(set(counts_a.index).union(counts_b.index))
    expected_agreement = sum(
        counts_a.get(label, 0.0) * counts_b.get(label, 0.0)
        for label in all_labels
    )

    if expected_agreement == 1.0:
        kappa = 1.0 if observed_agreement == 1.0 else 0.0
    else:
        kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)

    return kappa, observed_agreement, expected_agreement


def build_label_counts(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for data_type, group in merged.groupby("data_type", dropna=False):
        for model_name, label_col in (
            ("vsfc_phobert", "vsfc_phobert_label"),
            ("phobert_student_vn", "phobert_student_vn_label"),
        ):
            counts = group[label_col].dropna()
            counts = counts[counts != ""].value_counts().sort_index()
            for label, count in counts.items():
                rows.append(
                    {
                        "data_type": data_type,
                        "model": model_name,
                        "sentiment_label": label,
                        "count": int(count),
                    }
                )
    return pd.DataFrame(rows)


def build_report(merged: pd.DataFrame, diagnostics: list[dict]) -> dict:
    report: dict[str, object] = {
        "total_rows": int(len(merged)),
        "diagnostics": diagnostics,
        "by_data_type": {},
    }

    for data_type, group in merged.groupby("data_type", dropna=False):
        comparable = group[
            (group["vsfc_phobert_label"].fillna("") != "")
            & (group["phobert_student_vn_label"].fillna("") != "")
        ].copy()
        kappa, observed, expected = cohen_kappa(
            comparable["vsfc_phobert_label"],
            comparable["phobert_student_vn_label"],
        )
        confusion = pd.crosstab(
            comparable["vsfc_phobert_label"],
            comparable["phobert_student_vn_label"],
            rownames=["vsfc_phobert"],
            colnames=["phobert_student_vn"],
            dropna=False,
        )
        report["by_data_type"][str(data_type)] = {
            "rows": int(len(group)),
            "comparable_rows": int(len(comparable)),
            "agree_count": int((comparable["models_agree"] == True).sum()),
            "disagree_count": int((comparable["models_agree"] == False).sum()),
            "observed_agreement": observed,
            "expected_agreement": expected,
            "cohen_kappa": kappa,
            "merge_status_counts": group["merge_status"].value_counts().to_dict(),
            "confusion_matrix": confusion.to_dict(),
        }

    comparable_all = merged[
        (merged["vsfc_phobert_label"].fillna("") != "")
        & (merged["phobert_student_vn_label"].fillna("") != "")
    ].copy()
    kappa, observed, expected = cohen_kappa(
        comparable_all["vsfc_phobert_label"],
        comparable_all["phobert_student_vn_label"],
    )
    report["overall"] = {
        "comparable_rows": int(len(comparable_all)),
        "agree_count": int((comparable_all["models_agree"] == True).sum()),
        "disagree_count": int((comparable_all["models_agree"] == False).sum()),
        "observed_agreement": observed,
        "expected_agreement": expected,
        "cohen_kappa": kappa,
    }
    return report


def write_outputs(
    merged: pd.DataFrame,
    counts: pd.DataFrame,
    report: dict,
    output_merged: str | Path,
    output_counts: str | Path,
    output_report: str | Path,
) -> None:
    merged_path = Path(output_merged)
    counts_path = Path(output_counts)
    report_path = Path(output_report)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    counts.to_csv(counts_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge post/comment sentiment labels from VSFC PhoBERT and phobert_student_vn."
    )
    parser.add_argument("--comments_phobert_student_vn", default=str(DEFAULT_COMMENTS_phobert_student_vn))
    parser.add_argument("--posts_phobert_student_vn", default=str(DEFAULT_POSTS_phobert_student_vn))
    parser.add_argument("--posts_vsfc_phobert", default=str(DEFAULT_POSTS_VSFC_PHOBERT))
    parser.add_argument(
        "--comments_vsfc_phobert",
        default=str(DEFAULT_COMMENTS_VSFC_PHOBERT),
    )
    parser.add_argument("--output_merged", default=str(DEFAULT_OUTPUT_MERGED))
    parser.add_argument("--output_counts", default=str(DEFAULT_OUTPUT_COUNTS))
    parser.add_argument("--output_report", default=str(DEFAULT_OUTPUT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    comments, comments_diagnostics = merge_dataset(
        data_type="comment",
        phobert_student_vn_path=args.comments_phobert_student_vn,
        vsfc_phobert_path=args.comments_vsfc_phobert,
    )
    posts, posts_diagnostics = merge_dataset(
        data_type="post",
        phobert_student_vn_path=args.posts_phobert_student_vn,
        vsfc_phobert_path=args.posts_vsfc_phobert,
    )

    merged = pd.concat([posts, comments], ignore_index=True)
    counts = build_label_counts(merged)
    diagnostics = [posts_diagnostics, comments_diagnostics]
    report = build_report(merged, diagnostics)

    write_outputs(
        merged=merged,
        counts=counts,
        report=report,
        output_merged=args.output_merged,
        output_counts=args.output_counts,
        output_report=args.output_report,
    )

    print("=== Merge complete ===")
    print(f"Merged data : {args.output_merged}")
    print(f"Label counts: {args.output_counts}")
    print(f"Report      : {args.output_report}")
    print()
    for item in diagnostics:
        print(
            f"{item['data_type']}: merged={item['merged_rows']}, "
            f"both={item['both_models_rows']}, "
            f"only_phobert_student_vn={item['only_phobert_student_vn_rows']}, "
            f"only_vsfc_phobert={item['only_vsfc_phobert_rows']}"
        )


if __name__ == "__main__":
    main()
