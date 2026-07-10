from __future__ import annotations

import csv
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


BASE_DIR = Path(__file__).resolve().parent
RULE_PATH = BASE_DIR / "rule_label.md"
OUTPUT_PATH = BASE_DIR / "merch_datasets_student_sentiment.csv"

OUTPUT_COLUMNS = [
    "data_types",
    "record_key",
    "id",
    "content_anonymized",
    "emotion",
    "label",
]


@dataclass(frozen=True)
class DatasetConfig:
    source_name: str
    path: Path


DATASETS = [
    DatasetConfig("uit_vsmec_test", BASE_DIR / "UIT-VSMEC" / "test.csv"),
    DatasetConfig("uit_vsmec_train", BASE_DIR / "UIT-VSMEC" / "train.csv"),
    DatasetConfig("uit_vsmec_valid", BASE_DIR / "UIT-VSMEC" / "valid.csv"),
    DatasetConfig("combined_data_2_utc2", BASE_DIR / "combined_data_2_utc2.xlsx"),
    DatasetConfig(
        "final_student_dntu_senitment_datasets",
        BASE_DIR / "final_student_dntu_senitment_datasets.csv",
    ),
    DatasetConfig("train_utc2", BASE_DIR / "train_utc2.xlsx"),
]


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char)).lower()


def normalize_label_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def parse_rule_label(path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    label_to_emotion: dict[str, str] = {}
    emotion_to_label: dict[str, str] = {}
    raw_emotion_to_emotion: dict[str, str] = {}

    line_pattern = re.compile(r"^\s*(\d+)\s*,\s*([^,]+?)\s*,\s*\[(.*?)\]\s*$")
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = line_pattern.match(line)
        if not match:
            continue

        label, emotion, raw_values = match.groups()
        label = label.strip()
        emotion = emotion.strip()

        label_to_emotion[label] = emotion
        emotion_to_label[normalize_text(emotion)] = label
        raw_emotion_to_emotion[normalize_text(emotion)] = emotion

        for raw_value in raw_values.split(","):
            raw_emotion = raw_value.strip()
            if raw_emotion:
                raw_emotion_to_emotion[normalize_text(raw_emotion)] = emotion

    if not label_to_emotion:
        raise ValueError(f"Could not parse label rules from {path}")

    return label_to_emotion, emotion_to_label, raw_emotion_to_emotion


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


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    candidate_set = {normalize_text(candidate) for candidate in candidates}
    for column in df.columns:
        if normalize_text(column) in candidate_set:
            return str(column)
    return None


def resolve_emotion_and_label(
    value: object,
    source_column: str,
    label_to_emotion: dict[str, str],
    emotion_to_label: dict[str, str],
    raw_emotion_to_emotion: dict[str, str],
) -> tuple[str, str]:
    normalized_column = normalize_text(source_column)
    normalized_value = normalize_text(value)

    if normalized_column == "label":
        label = normalize_label_value(value)
        emotion = label_to_emotion.get(label)
        if emotion is None:
            raise ValueError(f"Unknown numeric label: {value!r}")
        return emotion, label

    emotion = raw_emotion_to_emotion.get(normalized_value)
    if emotion is None:
        emotion = value if isinstance(value, str) else str(value)
        emotion = emotion.strip()

    label = emotion_to_label.get(normalize_text(emotion))
    if label is None:
        raise ValueError(f"Unknown emotion/sentiment value: {value!r}")

    return emotion, label


def get_row_id(row: pd.Series, fallback_position: int) -> str:
    if "id" in row.index and not pd.isna(row["id"]) and str(row["id"]).strip():
        return normalize_label_value(row["id"])
    return str(fallback_position)


def get_data_type_and_record_key(
    row: pd.Series,
    source_name: str,
    row_id: str,
) -> tuple[str, str]:
    if (
        "data_types" in row.index
        and "record_key" in row.index
        and not pd.isna(row["data_types"])
        and not pd.isna(row["record_key"])
        and str(row["data_types"]).strip()
        and str(row["record_key"]).strip()
    ):
        return str(row["data_types"]).strip(), str(row["record_key"]).strip()

    return source_name, f"{source_name}:{row_id}"


def build_rows(
    config: DatasetConfig,
    label_to_emotion: dict[str, str],
    emotion_to_label: dict[str, str],
    raw_emotion_to_emotion: dict[str, str],
) -> list[dict[str, str]]:
    if not config.path.exists():
        raise FileNotFoundError(config.path)

    output_rows: list[dict[str, str]] = []
    tables = load_tables(config.path)

    for sheet_name, df in tables.items():
        text_col = find_column(df, ["content_anonymized", "sentences", "sentence", "content"])
        label_col = find_column(df, ["label", "sentiment", "emotion", "cảm xúc", "cam xuc"])

        if not text_col or not label_col:
            print(
                f"Skip {config.source_name}/{sheet_name}: "
                f"missing text or label-like column"
            )
            continue

        for index, row in df.iterrows():
            content = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
            if not content:
                raise ValueError(
                    f"Blank content in {config.source_name}/{sheet_name}, "
                    f"row {index + 2}"
                )

            emotion, label = resolve_emotion_and_label(
                row[label_col],
                label_col,
                label_to_emotion,
                emotion_to_label,
                raw_emotion_to_emotion,
            )

            row_id = get_row_id(row, fallback_position=index + 1)
            data_type, record_key = get_data_type_and_record_key(
                row,
                source_name=config.source_name,
                row_id=row_id,
            )
            output_rows.append(
                {
                    "data_types": data_type,
                    "record_key": record_key,
                    "id": row_id,
                    "content_anonymized": content,
                    "emotion": emotion,
                    "label": label,
                }
            )

        print(
            f"Loaded {config.source_name}/{sheet_name}: "
            f"{len(df)} rows -> {len(output_rows)} total rows"
        )

    return output_rows


def main() -> None:
    label_to_emotion, emotion_to_label, raw_emotion_to_emotion = parse_rule_label(
        RULE_PATH
    )

    all_rows: list[dict[str, str]] = []
    for config in DATASETS:
        all_rows.extend(
            build_rows(
                config,
                label_to_emotion,
                emotion_to_label,
                raw_emotion_to_emotion,
            )
        )

    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    result = pd.DataFrame(all_rows)
    print("=" * 90)
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows: {len(result)}")
    print(f"Columns: {OUTPUT_COLUMNS}")
    print("Rows by data_types:")
    print(result["data_types"].value_counts().sort_index().to_string())
    print("Rows by emotion:")
    print(result["emotion"].value_counts().sort_index().to_string())
    print("Rows by label:")
    print(result["label"].value_counts().sort_index().to_string())
    print(f"Blank content_anonymized rows: {result['content_anonymized'].eq('').sum()}")
    print(f"Blank label rows: {result['label'].eq('').sum()}")


if __name__ == "__main__":
    main()
