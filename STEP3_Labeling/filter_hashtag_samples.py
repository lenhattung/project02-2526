# -*- coding: utf-8 -*-
"""
Filter rows that contain real hashtags in content_anonymized.

Post-id markers such as #Cfs2295_DNTU, #Cfs2245_DNTU, or #24399 are ignored
because they are IDs written in the content, usually at the beginning.
Output keeps the exact same columns as the source CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_INPUT = Path(__file__).with_name("merch_datasets_student_sentiment.csv")
DEFAULT_OUTPUT = Path(__file__).with_name(
    "merch_datasets_student_sentiment_hashtag_samples.csv"
)

CONTENT_COLUMN = "content_anonymized"
HASHTAG_RE = re.compile(r"(?<!\w)#\w+", flags=re.UNICODE)

# Explicit examples from the request. The regexes below also cover the same
# style more generally.
EXACT_IGNORED_POST_ID_HASHTAGS = {
    "#cfs2295_dntu",
    "#cfs2245_dntu",
    "#24399",
}

IGNORED_POST_ID_HASHTAG_PATTERNS = (
    re.compile(r"^#cfs\d+_dntu", flags=re.IGNORECASE),
    re.compile(r"^#\d+$"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export rows containing hashtags, while ignoring post-id "
            "markers such as #Cfs2295_DNTU, #Cfs2245_DNTU, or #24399."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source CSV path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Filtered CSV path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--content-column",
        default=CONTENT_COLUMN,
        help=f"Text column to scan for hashtags. Default: {CONTENT_COLUMN}",
    )
    return parser.parse_args()


def is_ignored_post_id_hashtag(hashtag: str) -> bool:
    normalized = hashtag.lower()
    if normalized in EXACT_IGNORED_POST_ID_HASHTAGS:
        return True
    return any(pattern.match(hashtag) for pattern in IGNORED_POST_ID_HASHTAG_PATTERNS)


def has_real_hashtag(text: str) -> tuple[bool, bool]:
    """Return (has_real_hashtag, has_any_hashtag)."""
    has_any_hashtag = False
    for match in HASHTAG_RE.finditer(text or ""):
        has_any_hashtag = True
        if not is_ignored_post_id_hashtag(match.group(0)):
            return True, has_any_hashtag
    return False, has_any_hashtag


def export_hashtag_samples(input_path: Path, output_path: Path, content_column: str) -> None:
    total_rows = 0
    rows_with_any_hashtag = 0
    exported_rows = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise ValueError(f"Input file has no header: {input_path}")
        if content_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{content_column}' not found. Available columns: "
                f"{', '.join(reader.fieldnames)}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8-sig", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                total_rows += 1
                keep_row, has_any_hashtag = has_real_hashtag(row.get(content_column, ""))
                if has_any_hashtag:
                    rows_with_any_hashtag += 1
                if keep_row:
                    writer.writerow(row)
                    exported_rows += 1

    ignored_only_rows = rows_with_any_hashtag - exported_rows
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total rows: {total_rows}")
    print(f"Rows with any hashtag: {rows_with_any_hashtag}")
    print(f"Rows exported: {exported_rows}")
    print(f"Rows containing only ignored post-id hashtags: {ignored_only_rows}")


def main() -> None:
    args = parse_args()
    export_hashtag_samples(args.input, args.output, args.content_column)


if __name__ == "__main__":
    main()
