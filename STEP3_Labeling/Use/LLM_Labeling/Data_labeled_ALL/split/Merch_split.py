# -*- coding: utf-8 -*-
"""
Merge split comment label files by id.

Sources:
    - data/*.csv: GPT5.5_Medium labels
    - Gemini/*_labeled.xlsx: 3.5 Flash labels
    - Gemini/*_labeled_v2.xlsx: 3.1 Pro labels

Output columns:
    id, content_anonymized, 3.5 Flash, 3.1 Pro, GPT5.5_Medium

Example:
    python STEP3_Labeling/LLM_Labeling/Data_labeled/split/Merch_split.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GPT_DIR = SCRIPT_DIR / "data"
DEFAULT_GEMINI_DIR = SCRIPT_DIR / "Gemini"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "comments_dataset_labeled_merged.csv"

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


def list_input_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls"}
    )
    if not files:
        raise FileNotFoundError(f"No CSV/XLSX files found in {folder}")
    return files


def require_columns(df: pd.DataFrame, columns: list[str], path: str | Path) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing columns in {path}: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )


def require_unique_id(df: pd.DataFrame, source_name: str) -> None:
    duplicated_count = int(df["id"].duplicated().sum())
    if duplicated_count:
        raise ValueError(f"{source_name} has {duplicated_count} duplicated id values")


def normalize_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="raise").astype("int64")
    return df


def load_gpt_labels(folder: str | Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in list_input_files(folder):
        df = read_table(path)
        require_columns(df, ["id", "content_anonymized"], path)

        if "GPT5.5_Medium" in df.columns:
            label_col = "GPT5.5_Medium"
        elif "label" in df.columns:
            label_col = "label"
        else:
            raise ValueError(
                f"Missing GPT label column in {path}. "
                "Expected GPT5.5_Medium or label."
            )

        frame = df[["id", "content_anonymized", label_col]].copy()
        frame = frame.rename(columns={label_col: "GPT5.5_Medium"})
        frames.append(frame)

    combined = normalize_id(pd.concat(frames, ignore_index=True))
    require_unique_id(combined, "GPT5.5_Medium files")
    return combined


def gemini_model_column(path: Path) -> str:
    stem = path.stem.lower()
    if "_v2" in stem:
        return "3.1 Pro"
    return "3.5 Flash"


def load_gemini_labels(folder: str | Path) -> pd.DataFrame:
    grouped_frames: dict[str, list[pd.DataFrame]] = {
        "3.5 Flash": [],
        "3.1 Pro": [],
    }

    for path in list_input_files(folder):
        df = read_table(path)
        require_columns(df, ["id", "content_anonymized", "label"], path)
        model_col = gemini_model_column(path)
        frame = df[["id", "content_anonymized", "label"]].copy()
        frame = frame.rename(columns={"label": model_col})
        grouped_frames[model_col].append(frame)

    model_dfs: list[pd.DataFrame] = []
    for model_col, frames in grouped_frames.items():
        if not frames:
            raise FileNotFoundError(f"No Gemini files found for {model_col}")

        combined = normalize_id(pd.concat(frames, ignore_index=True))
        require_unique_id(combined, f"Gemini {model_col} files")
        model_dfs.append(combined)

    merged = model_dfs[0].merge(
        model_dfs[1],
        on="id",
        how="outer",
        suffixes=("_35", "_31"),
        validate="one_to_one",
    )
    merged["content_anonymized"] = merged["content_anonymized_35"].combine_first(
        merged["content_anonymized_31"]
    )
    return merged[["id", "content_anonymized", "3.5 Flash", "3.1 Pro"]]


def merge_split_labels(
    gpt_dir: str | Path = DEFAULT_GPT_DIR,
    gemini_dir: str | Path = DEFAULT_GEMINI_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    gpt_df = load_gpt_labels(gpt_dir)
    gemini_df = load_gemini_labels(gemini_dir)

    merged = gemini_df.merge(
        gpt_df,
        on="id",
        how="outer",
        suffixes=("_gemini", "_gpt"),
        validate="one_to_one",
        indicator=True,
    )

    missing_counts = merged["_merge"].value_counts().to_dict()
    if missing_counts.get("left_only", 0) or missing_counts.get("right_only", 0):
        print(f"Warning: ID coverage mismatch: {missing_counts}")

    content_diff = (
        merged["content_anonymized_gemini"].fillna("").astype(str)
        != merged["content_anonymized_gpt"].fillna("").astype(str)
    )
    both_sides = merged["_merge"] == "both"
    if (content_diff & both_sides).any():
        print(
            "Warning: "
            f"{int((content_diff & both_sides).sum())} matched rows have different content_anonymized values."
        )

    merged["content_anonymized"] = merged["content_anonymized_gemini"].combine_first(
        merged["content_anonymized_gpt"]
    )
    result = merged[FINAL_COLUMNS].sort_values("id").reset_index(drop=True)
    write_table(result, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge split comment labels by id.")
    parser.add_argument("--gpt_dir", default=str(DEFAULT_GPT_DIR), help="Folder with GPT5.5_Medium files")
    parser.add_argument("--gemini_dir", default=str(DEFAULT_GEMINI_DIR), help="Folder with Gemini files")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Merged output path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merged_df = merge_split_labels(
        gpt_dir=args.gpt_dir,
        gemini_dir=args.gemini_dir,
        output_path=args.output,
    )
    print(f"Saved {len(merged_df)} rows to {args.output}")
