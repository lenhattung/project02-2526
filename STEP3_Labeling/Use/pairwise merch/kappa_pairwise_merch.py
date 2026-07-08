# -*- coding: utf-8 -*-
"""
Build pairwise Cohen's Kappa for all merchant sentiment model columns.

Default usage:
    python "STEP3_Labeling/pairwise merch/kappa_pairwise_merch.py"

Inputs:
    - STEP3_Labeling/kappa_merged_all_model_training.csv
    - STEP3_Labeling/LLM_Labeling/Data_LLM_Labeled_Merch/llm_kappa_merged_all.csv

Outputs:
    - kappa_pairwise_merch_merged_by_id.csv
    - kappa_pairwise_merch.csv
    - kappa_pairwise_merch_label_counts.csv
    - kappa_pairwise_merch_report.json

Labels are normalized to:
    0 = negative, 1 = neutral, 2 = positive
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STEP3_DIR = SCRIPT_DIR.parent

DEFAULT_TRAINING_PATH = STEP3_DIR / "kappa_merged_all_model_training.csv"
DEFAULT_LLM_PATH = (
    STEP3_DIR
    / "LLM_Labeling"
    / "Data_LLM_Labeled_Merch"
    / "llm_kappa_merged_all.csv"
)

DEFAULT_OUTPUT_MERGED = SCRIPT_DIR / "kappa_pairwise_merch_merged_by_id.csv"
DEFAULT_OUTPUT_PAIRWISE = SCRIPT_DIR / "kappa_pairwise_merch.csv"
DEFAULT_OUTPUT_COUNTS = SCRIPT_DIR / "kappa_pairwise_merch_label_counts.csv"
DEFAULT_OUTPUT_REPORT = SCRIPT_DIR / "kappa_pairwise_merch_report.json"

ID_COLUMN = "id"
DATA_TYPE_COLUMN = "data_type"
RECORD_KEY_COLUMN = "record_key"
TEXT_COLUMN = "content_anonymized"

TRAINING_MODEL_COLUMNS = {
    "vsfc_phobert": "vsfc_phobert_label",
    "phobert_student_vn": "phobert_student_vn_label",
}
LLM_MODEL_COLUMNS = {
    "3.5 Flash": "3.5 Flash",
    "3.1 Pro": "3.1 Pro",
    "GPT5.5_Medium": "GPT5.5_Medium",
}
LLM_MAJORITY_SOURCE_COLUMN = "label"
LLM_MAJORITY_OUTPUT_COLUMN = "llm_majority_label"

PAIRWISE_MODEL_COLUMNS = tuple(TRAINING_MODEL_COLUMNS) + tuple(LLM_MODEL_COLUMNS)
ALL_LABEL_COLUMNS = (*PAIRWISE_MODEL_COLUMNS, LLM_MAJORITY_OUTPUT_COLUMN)

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


def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = " ".join(str(value).strip().split())
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    label = " ".join(str(value).strip().split())
    if not label:
        return ""
    return LABEL_NORMALIZATION.get(label, LABEL_NORMALIZATION.get(label.upper(), label))


def require_columns(df: pd.DataFrame, columns: list[str], file_label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{file_label} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def ensure_unique_keys(df: pd.DataFrame, key_columns: list[str], file_label: str) -> None:
    duplicate_mask = df.duplicated(key_columns, keep=False)
    if not duplicate_mask.any():
        return

    duplicates = df.loc[duplicate_mask, key_columns].head(10).to_dict("records")
    raise ValueError(
        f"{file_label} has duplicate merge keys {key_columns}. "
        f"First duplicate keys: {duplicates}"
    )


def prepare_training(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        df,
        [ID_COLUMN, DATA_TYPE_COLUMN, *TRAINING_MODEL_COLUMNS.values()],
        "training file",
    )

    base_columns = [
        column
        for column in (
            DATA_TYPE_COLUMN,
            RECORD_KEY_COLUMN,
            ID_COLUMN,
            TEXT_COLUMN,
            "content",
            "source",
            "posted_at",
        )
        if column in df.columns
    ]
    result = df[base_columns].copy()
    result["_id_key"] = df[ID_COLUMN].map(normalize_id)
    result["_row_order"] = range(len(result))

    for output_column, source_column in TRAINING_MODEL_COLUMNS.items():
        result[output_column] = df[source_column].map(normalize_label)

    return result


def prepare_llm(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, [ID_COLUMN, *LLM_MODEL_COLUMNS.values()], "LLM file")

    base_columns = [ID_COLUMN]
    if DATA_TYPE_COLUMN in df.columns:
        base_columns.insert(0, DATA_TYPE_COLUMN)
    if RECORD_KEY_COLUMN in df.columns:
        base_columns.insert(1, RECORD_KEY_COLUMN)
    if TEXT_COLUMN in df.columns:
        base_columns.append(TEXT_COLUMN)

    result = df[base_columns].copy()
    result["_id_key"] = df[ID_COLUMN].map(normalize_id)
    result["_row_order"] = range(len(result))

    for output_column, source_column in LLM_MODEL_COLUMNS.items():
        result[output_column] = df[source_column].map(normalize_label)

    if LLM_MAJORITY_SOURCE_COLUMN in df.columns:
        result[LLM_MAJORITY_OUTPUT_COLUMN] = df[LLM_MAJORITY_SOURCE_COLUMN].map(
            normalize_label
        )
    else:
        result[LLM_MAJORITY_OUTPUT_COLUMN] = ""

    return result


def attach_training_keys_to_llm(
    training: pd.DataFrame, llm: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    if DATA_TYPE_COLUMN in llm.columns:
        return llm, {"llm_key_alignment": "llm_file_had_data_type"}

    same_length = len(training) == len(llm)
    same_id_order = (
        same_length
        and training["_id_key"].reset_index(drop=True).equals(
            llm["_id_key"].reset_index(drop=True)
        )
    )

    if same_id_order:
        llm = llm.copy()
        llm[DATA_TYPE_COLUMN] = training[DATA_TYPE_COLUMN].to_numpy()
        if RECORD_KEY_COLUMN in training.columns:
            llm[RECORD_KEY_COLUMN] = training[RECORD_KEY_COLUMN].to_numpy()
        return llm, {
            "llm_key_alignment": "copied_data_type_from_training_row_order",
            "row_order_id_matches": True,
        }

    training_keys = training[[DATA_TYPE_COLUMN, "_id_key"]].copy()
    if RECORD_KEY_COLUMN in training.columns:
        training_keys[RECORD_KEY_COLUMN] = training[RECORD_KEY_COLUMN]
    training_keys["_id_occurrence"] = training_keys.groupby("_id_key").cumcount()

    llm = llm.copy()
    llm["_id_occurrence"] = llm.groupby("_id_key").cumcount()
    llm = llm.merge(
        training_keys,
        on=["_id_key", "_id_occurrence"],
        how="left",
        validate="one_to_one",
    )
    missing_data_type = int(llm[DATA_TYPE_COLUMN].isna().sum())
    return llm, {
        "llm_key_alignment": "copied_data_type_from_training_id_occurrence",
        "row_order_id_matches": False,
        "llm_rows_without_training_key": missing_data_type,
    }


def merge_inputs(training: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    key_columns = [DATA_TYPE_COLUMN, "_id_key"]
    ensure_unique_keys(training, key_columns, "training file")
    ensure_unique_keys(llm, key_columns, "LLM file")

    llm_side_columns = [
        DATA_TYPE_COLUMN,
        "_id_key",
        *LLM_MODEL_COLUMNS.keys(),
        LLM_MAJORITY_OUTPUT_COLUMN,
    ]
    if TEXT_COLUMN in llm.columns and TEXT_COLUMN not in training.columns:
        llm_side_columns.append(TEXT_COLUMN)

    merged = training.merge(
        llm[llm_side_columns],
        on=key_columns,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    merged["merge_status"] = merged["_merge"].map(
        {
            "both": "both_input_files",
            "left_only": "only_training_file",
            "right_only": "only_llm_file",
        }
    )
    if ID_COLUMN not in merged.columns:
        merged[ID_COLUMN] = merged["_id_key"]
    else:
        merged[ID_COLUMN] = merged[ID_COLUMN].fillna(merged["_id_key"])

    ordered_columns = [
        DATA_TYPE_COLUMN,
        RECORD_KEY_COLUMN,
        ID_COLUMN,
        TEXT_COLUMN,
        *PAIRWISE_MODEL_COLUMNS,
        LLM_MAJORITY_OUTPUT_COLUMN,
        "merge_status",
        "content",
        "source",
        "posted_at",
    ]
    ordered_columns = [column for column in ordered_columns if column in merged.columns]
    remaining_columns = [
        column
        for column in merged.columns
        if column not in ordered_columns and not column.startswith("_")
    ]
    return merged[ordered_columns + remaining_columns]


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


def build_pairwise_report(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pair_rows: list[dict] = []
    report: dict[str, object] = {
        "total_rows": int(len(merged)),
        "model_columns": list(PAIRWISE_MODEL_COLUMNS),
        "label_mapping": {"0": "negative", "1": "neutral", "2": "positive"},
        "merge_status_counts": merged["merge_status"].value_counts().to_dict(),
        "by_data_type": {},
        "overall": {},
    }

    for data_type, group in merged.groupby(DATA_TYPE_COLUMN, dropna=False):
        data_type_key = str(data_type)
        report["by_data_type"][data_type_key] = {}
        for model_a, model_b in combinations(PAIRWISE_MODEL_COLUMNS, 2):
            item = compare_pair(group, data_type_key, model_a, model_b)
            report["by_data_type"][data_type_key][f"{model_a} vs {model_b}"] = item
            pair_rows.append(
                {key: value for key, value in item.items() if key != "confusion_matrix"}
            )

    for model_a, model_b in combinations(PAIRWISE_MODEL_COLUMNS, 2):
        item = compare_pair(merged, "overall", model_a, model_b)
        report["overall"][f"{model_a} vs {model_b}"] = item
        pair_rows.append(
            {key: value for key, value in item.items() if key != "confusion_matrix"}
        )

    return pd.DataFrame(pair_rows), report


def build_label_counts(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    groups = [(str(data_type), group) for data_type, group in merged.groupby(DATA_TYPE_COLUMN)]
    groups.append(("overall", merged))

    for data_type, group in groups:
        for model_name in ALL_LABEL_COLUMNS:
            if model_name not in group.columns:
                continue
            labels = group[model_name].dropna().astype(str).str.strip()
            labels = labels[labels != ""].value_counts().sort_index()
            for label, count in labels.items():
                rows.append(
                    {
                        "data_type": data_type,
                        "model": model_name,
                        "label": label,
                        "label_name": {"0": "negative", "1": "neutral", "2": "positive"}.get(
                            label, ""
                        ),
                        "count": int(count),
                    }
                )

    return pd.DataFrame(rows)


def write_outputs(
    merged: pd.DataFrame,
    pairwise: pd.DataFrame,
    counts: pd.DataFrame,
    report: dict,
    output_merged: str | Path,
    output_pairwise: str | Path,
    output_counts: str | Path,
    output_report: str | Path,
) -> None:
    output_merged = Path(output_merged)
    output_pairwise = Path(output_pairwise)
    output_counts = Path(output_counts)
    output_report = Path(output_report)

    output_merged.parent.mkdir(parents=True, exist_ok=True)
    output_pairwise.parent.mkdir(parents=True, exist_ok=True)
    output_counts.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(output_merged, index=False, encoding="utf-8-sig")
    pairwise.to_csv(output_pairwise, index=False, encoding="utf-8-sig")
    counts.to_csv(output_counts, index=False, encoding="utf-8-sig")
    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(
    training_path: str | Path = DEFAULT_TRAINING_PATH,
    llm_path: str | Path = DEFAULT_LLM_PATH,
    output_merged: str | Path = DEFAULT_OUTPUT_MERGED,
    output_pairwise: str | Path = DEFAULT_OUTPUT_PAIRWISE,
    output_counts: str | Path = DEFAULT_OUTPUT_COUNTS,
    output_report: str | Path = DEFAULT_OUTPUT_REPORT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    training_raw = read_table(training_path)
    llm_raw = read_table(llm_path)

    training = prepare_training(training_raw)
    llm = prepare_llm(llm_raw)
    llm, alignment_report = attach_training_keys_to_llm(training, llm)

    merged = merge_inputs(training, llm)
    pairwise, report = build_pairwise_report(merged)
    report["input_rows"] = {
        "training": int(len(training_raw)),
        "llm": int(len(llm_raw)),
    }
    report.update(alignment_report)
    counts = build_label_counts(merged)

    write_outputs(
        merged=merged,
        pairwise=pairwise,
        counts=counts,
        report=report,
        output_merged=output_merged,
        output_pairwise=output_pairwise,
        output_counts=output_counts,
        output_report=output_report,
    )

    return merged, pairwise, counts, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pairwise Cohen's Kappa across training and LLM merchant models."
    )
    parser.add_argument("--training", default=str(DEFAULT_TRAINING_PATH))
    parser.add_argument("--llm", default=str(DEFAULT_LLM_PATH))
    parser.add_argument("--output_merged", default=str(DEFAULT_OUTPUT_MERGED))
    parser.add_argument("--output_pairwise", default=str(DEFAULT_OUTPUT_PAIRWISE))
    parser.add_argument("--output_counts", default=str(DEFAULT_OUTPUT_COUNTS))
    parser.add_argument("--output_report", default=str(DEFAULT_OUTPUT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged, pairwise, counts, report = run(
        training_path=args.training,
        llm_path=args.llm,
        output_merged=args.output_merged,
        output_pairwise=args.output_pairwise,
        output_counts=args.output_counts,
        output_report=args.output_report,
    )

    print("=== Merchant pairwise Kappa complete ===")
    print(f"Merged data : {args.output_merged}")
    print(f"Pairwise    : {args.output_pairwise}")
    print(f"Label counts: {args.output_counts}")
    print(f"Report      : {args.output_report}")
    print()
    print(f"Total rows  : {len(merged)}")
    print(f"Alignment   : {report.get('llm_key_alignment')}")
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
