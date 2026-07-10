from pathlib import Path
import sys
import unicodedata

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent

FILES = [
    BASE_DIR / "UIT-VSMEC" / "test.csv",
    BASE_DIR / "UIT-VSMEC" / "train.csv",
    BASE_DIR / "UIT-VSMEC" / "valid.csv",
    BASE_DIR / "combined_data_2.xlsx",
    BASE_DIR / "final_student_dntu_senitment_datasets.csv",
    BASE_DIR / "train_utc2.xlsx",
]

TARGET_COLUMNS = {"emotion", "label", "sentiment", "cam xuc"}


def normalize_column_name(value: object) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def load_tables(path: Path) -> dict[str, pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return {"csv": read_csv(path)}
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=None)
    raise ValueError(f"Unsupported file type: {path}")


def find_target_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if normalize_column_name(column) in TARGET_COLUMNS
    ]


def print_value_counts(df: pd.DataFrame, column: str) -> None:
    series = df[column]
    missing_count = int(series.isna().sum() + series.astype(str).str.strip().eq("").sum())
    display_series = series.map(
        lambda value: "<NA>" if pd.isna(value) else str(value).strip() or "<BLANK>"
    )
    counts = display_series.value_counts(dropna=False).sort_index()

    print(f"  Column: {column}")
    print(f"  Unique values: {series.nunique(dropna=True)}")
    print(f"  Missing/blank rows: {missing_count}")
    print("  Value counts:")
    for value, count in counts.items():
        print(f"    {value!r}: {count}")


def inspect_file(path: Path) -> None:
    print("=" * 90)
    print(f"File: {path}")

    if not path.exists():
        print("Status: NOT FOUND")
        return

    tables = load_tables(path)
    for sheet_name, df in tables.items():
        print("-" * 90)
        print(f"Sheet/source: {sheet_name}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")

        target_columns = find_target_columns(df)
        if not target_columns:
            print("No emotion/label/sentiment columns found.")
            continue

        for column in target_columns:
            print_value_counts(df, column)


def main() -> None:
    for path in FILES:
        inspect_file(path)


if __name__ == "__main__":
    main()
