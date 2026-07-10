# -*- coding: utf-8 -*-
"""Preprocess merch/student sentiment text data.

The output keeps every original column and adds `content_preprocessed`.
"""

from __future__ import annotations

import argparse
import ast
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PREPROCESSING_DIR = Path(__file__).resolve().parent
STEP3_DIR = PREPROCESSING_DIR.parent

DEFAULT_INPUT_CSV = STEP3_DIR / "merch_datasets_student_sentiment.csv"
DEFAULT_OUTPUT_CSV = PREPROCESSING_DIR / "merch_datasets_student_sentiment_preprocessed.csv"
DEFAULT_TEENCODE_PATH = PREPROCESSING_DIR / "teencode.txt"
DEFAULT_HASHTAG_MAP_PATH = PREPROCESSING_DIR / "hashtag_remove_map_student_sentiment.txt"

CONTENT_COLUMN = "content_anonymized"
OUTPUT_COLUMN = "content_preprocessed"
HEART_EMOTICON_PLACEHOLDER = "heartemoticonplaceholder"

URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|(?:[a-z0-9][a-z0-9._-]+\.(?:com|vn|edu|net|org|io|co)(?:/\S*)?)",
    flags=re.IGNORECASE,
)
HASHTAG_RE = re.compile(r"#\w+", flags=re.UNICODE)
ANGLE_SECTION_RE = re.compile(r"<([^<>]*)>")
VALID_ANGLE_TAG_RE = re.compile(r"[a-z]+(?:_[a-z]+)*")
REPEATED_ALPHA_RE = re.compile(r"([^\W\d_])\1{2,}", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
PERCENT_RE = re.compile(r"(?<!\w)(\d+)\s*%")
K_AMOUNT_RE = re.compile(r"(?<!\w)\d+k\b", flags=re.IGNORECASE)
NUMBER_SUFFIX_RE = re.compile(r"(?<!\w)(\d+)([kKmMbB])\b")
TIME_RE = re.compile(r"(?<!\w)(\d{1,2})h(\d{1,2})(?!\w)", flags=re.IGNORECASE)
HEIGHT_RE = re.compile(r"(?<!\w)(\d+)m(\d+)(?!\w)", flags=re.IGNORECASE)
NUMBER_UNIT_RE = re.compile(r"(?<!\w)(\d+)([a-zà-ỹđ]+)(?!\w)", flags=re.IGNORECASE)
LETTER_NUMBER_RE = re.compile(r"(?<!\w)([a-zà-ỹđ]+)(\d+)(?!\w)", flags=re.IGNORECASE)
MIXED_ALNUM_RE = re.compile(
    r"(?<!\w)(?=[a-zà-ỹđ0-9]*\d)(?=[a-zà-ỹđ0-9]*[a-zà-ỹđ])[a-zà-ỹđ0-9]+(?!\w)",
    flags=re.IGNORECASE,
)
DECIMAL_RE = re.compile(r"(?<!\w)(\d+)[,.](\d+)(?!\w)")
INTEGER_RE = re.compile(r"(?<!\w)\d+(?!\w)")

try:
    import emoji as emoji_lib
except ImportError:  # pragma: no cover - depends on local environment
    emoji_lib = None


DIGIT_WORDS = [
    "không",
    "một",
    "hai",
    "ba",
    "bốn",
    "năm",
    "sáu",
    "bảy",
    "tám",
    "chín",
]

TAG_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

UNIT_SUFFIX_MAP = {
    "k": "k",
    "m": "m",
    "b": "b",
    "tr": "triệu",
    "trieu": "triệu",
    "ty": "tỷ",
    "đ": "đồng",
    "d": "điểm",
    "vnd": "đồng",
    "t": "tuổi",
    "p": "phút",
    "ph": "phút",
    "h": "giờ",
    "g": "giờ",
    "s": "giây",
    "kg": "kg",
    "gr": "gram",
    "km": "km",
    "cm": "cm",
    "mm": "mm",
    "gb": "gb",
    "mb": "mb",
    "kb": "kb",
}

EMOJI_FALLBACK_NAMES = {
    "😀": "grinning_face",
    "😃": "grinning_face_with_big_eyes",
    "😄": "grinning_face_with_smiling_eyes",
    "😁": "beaming_face_with_smiling_eyes",
    "😆": "grinning_squinting_face",
    "😅": "grinning_face_with_sweat",
    "🤣": "rolling_on_the_floor_laughing",
    "😂": "face_with_tears_of_joy",
    "🙂": "slightly_smiling_face",
    "🙃": "upside_down_face",
    "😉": "winking_face",
    "😊": "smiling_face_with_smiling_eyes",
    "😇": "smiling_face_with_halo",
    "🥰": "smiling_face_with_hearts",
    "😍": "smiling_face_with_heart_eyes",
    "😘": "face_blowing_a_kiss",
    "😗": "kissing_face",
    "😚": "kissing_face_with_closed_eyes",
    "😋": "face_savoring_food",
    "😛": "face_with_tongue",
    "😜": "winking_face_with_tongue",
    "😝": "squinting_face_with_tongue",
    "🤪": "zany_face",
    "🤨": "face_with_raised_eyebrow",
    "🤔": "thinking_face",
    "🤭": "face_with_hand_over_mouth",
    "🤗": "hugging_face",
    "🤩": "star_struck",
    "😎": "smiling_face_with_sunglasses",
    "😏": "smirking_face",
    "😒": "unamused_face",
    "🙄": "face_with_rolling_eyes",
    "😬": "grimacing_face",
    "😑": "expressionless_face",
    "😐": "neutral_face",
    "😶": "face_without_mouth",
    "😌": "relieved_face",
    "😔": "pensive_face",
    "😪": "sleepy_face",
    "😴": "sleeping_face",
    "😷": "face_with_medical_mask",
    "🤒": "face_with_thermometer",
    "🤕": "face_with_head_bandage",
    "🤢": "nauseated_face",
    "🤮": "face_vomiting",
    "🤧": "sneezing_face",
    "🥲": "smiling_face_with_tear",
    "🥹": "face_holding_back_tears",
    "🥺": "pleading_face",
    "😢": "crying_face",
    "😭": "loudly_crying_face",
    "😤": "face_with_steam_from_nose",
    "😠": "angry_face",
    "😡": "enraged_face",
    "😳": "flushed_face",
    "😱": "face_screaming_in_fear",
    "😨": "fearful_face",
    "😰": "anxious_face_with_sweat",
    "😥": "sad_but_relieved_face",
    "😓": "downcast_face_with_sweat",
    "😩": "weary_face",
    "😞": "disappointed_face",
    "🙁": "slightly_frowning_face",
    "☹": "frowning_face",
    "☺": "smiling_face",
    "👍": "thumbs_up",
    "👎": "thumbs_down",
    "👌": "ok_hand",
    "👏": "clapping_hands",
    "🙏": "folded_hands",
    "💪": "flexed_biceps",
    "🙌": "raising_hands",
    "🙋": "person_raising_hand",
    "🤦": "person_facepalming",
    "👉": "backhand_index_pointing_right",
    "☝": "index_pointing_up",
    "✋": "raised_hand",
    "✌": "victory_hand",
    "🫶": "heart_hands",
    "❤": "red_heart",
    "❤️": "red_heart",
    "♥": "heart_suit",
    "💕": "two_hearts",
    "💖": "sparkling_heart",
    "💗": "growing_heart",
    "💙": "blue_heart",
    "💚": "green_heart",
    "💛": "yellow_heart",
    "💜": "purple_heart",
    "🖤": "black_heart",
    "🤍": "white_heart",
    "🔥": "fire",
    "✨": "sparkles",
    "⭐": "star",
    "🌟": "glowing_star",
    "🎉": "party_popper",
    "🎁": "wrapped_gift",
    "💯": "hundred_points",
    "✅": "check_mark_button",
    "✔": "check_mark",
    "❌": "cross_mark",
    "⚠": "warning",
    "‼": "double_exclamation_mark",
    "⁉": "exclamation_question_mark",
    "📍": "round_pushpin",
    "📌": "pushpin",
    "📞": "telephone_receiver",
    "☎": "telephone",
    "⏰": "alarm_clock",
    "🌸": "cherry_blossom",
    "🍀": "four_leaf_clover",
    "⚡": "high_voltage",
    "⚀": "die_face_one",
    "⚁": "die_face_two",
    "⚂": "die_face_three",
    "⚃": "die_face_four",
    "⚄": "die_face_five",
    "⚅": "die_face_six",
    "☀": "sun",
    "♂": "male_sign",
    "♀": "female_sign",
    "🏻": "light_skin_tone",
    "🏼": "medium_light_skin_tone",
    "🏽": "medium_skin_tone",
    "🏾": "medium_dark_skin_tone",
    "🏿": "dark_skin_tone",
}


@dataclass(frozen=True)
class PreprocessResources:
    teencode_map: dict[str, str]
    teencode_pattern: re.Pattern[str] | None
    hashtag_remove_regex: tuple[re.Pattern[str], ...]
    hashtag_remove_normalized: frozenset[str]


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", unicodedata.normalize("NFKC", str(text)))


def normalize_spaces(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize_tag_name(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    name = re.sub(
        r"\d",
        lambda match: "_" + TAG_DIGIT_WORDS[match.group(0)] + "_",
        name,
    )
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def normalize_hashtag_key(hashtag: str) -> str:
    hashtag = hashtag.strip().lstrip("#").lower()
    hashtag = strip_accents(unicodedata.normalize("NFC", hashtag))
    return hashtag


def maybe_fix_mojibake(text: str) -> str:
    if not any(marker in text for marker in ("Ã", "Â", "á»", "Ä", "Æ")):
        return text
    try:
        fixed = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text
    return fixed if fixed.count("�") <= text.count("�") else text


def read_text_file(path: Path) -> str:
    return maybe_fix_mojibake(path.read_text(encoding="utf-8-sig"))


def load_teencode_map(path: Path = DEFAULT_TEENCODE_PATH) -> dict[str, str]:
    teencode_map: dict[str, str] = {}
    for raw_line in read_text_file(path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            source, target = line.split("\t", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            source, target = parts
        source = normalize_unicode(source).lower().strip()
        target = normalize_unicode(target).lower().strip()
        if source and target:
            teencode_map[source] = target
    return teencode_map


def build_teencode_pattern(teencode_map: dict[str, str]) -> re.Pattern[str] | None:
    keys = sorted(teencode_map, key=len, reverse=True)
    if not keys:
        return None
    alternatives = "|".join(re.escape(key) for key in keys)
    return re.compile(rf"(?<!\w)({alternatives})(?!\w)", flags=re.UNICODE)


def load_hashtag_map(path: Path = DEFAULT_HASHTAG_MAP_PATH) -> tuple[tuple[re.Pattern[str], ...], frozenset[str]]:
    text = read_text_file(path)
    tree = ast.parse(text, filename=str(path))
    assignments: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.startswith("HASHTAG_"):
                assignments[target.id] = ast.literal_eval(node.value)

    regex_values = assignments.get("HASHTAG_REMOVE_REGEX", [])
    normalized_map = assignments.get("HASHTAG_REMOVE_MAP_NORMALIZED", {})
    exact_map = assignments.get("HASHTAG_REMOVE_MAP_EXACT", {})

    hashtag_remove_regex = tuple(
        re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)
        for pattern in regex_values
    )

    remove_normalized: set[str] = set()
    if isinstance(normalized_map, dict):
        remove_normalized.update(normalize_hashtag_key(str(key)) for key in normalized_map)
    if isinstance(exact_map, dict):
        remove_normalized.update(normalize_hashtag_key(str(key)) for key in exact_map)

    return hashtag_remove_regex, frozenset(remove_normalized)


def load_resources(
    teencode_path: Path = DEFAULT_TEENCODE_PATH,
    hashtag_map_path: Path = DEFAULT_HASHTAG_MAP_PATH,
) -> PreprocessResources:
    teencode_map = load_teencode_map(teencode_path)
    hashtag_remove_regex, hashtag_remove_normalized = load_hashtag_map(hashtag_map_path)
    return PreprocessResources(
        teencode_map=teencode_map,
        teencode_pattern=build_teencode_pattern(teencode_map),
        hashtag_remove_regex=hashtag_remove_regex,
        hashtag_remove_normalized=hashtag_remove_normalized,
    )


def sanitize_demojized_tags(text: str) -> str:
    return re.sub(
        r"<([^<>]+)>",
        lambda match: f"<{normalize_tag_name(match.group(1))}>",
        text,
    )


def normalize_angle_sections(text: str) -> str:
    def replace_angle_section(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if VALID_ANGLE_TAG_RE.fullmatch(inner):
            return f"<{inner}>"
        return " " + inner + " "

    previous = None
    while previous != text:
        previous = text
        text = ANGLE_SECTION_RE.sub(replace_angle_section, text)
    return text


def is_probably_emoji(ch: str) -> bool:
    codepoint = ord(ch)
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or 0x2300 <= codepoint <= 0x23FF
        or 0x2B00 <= codepoint <= 0x2BFF
        or codepoint in {0x203C, 0x2049, 0x2122, 0x2139, 0x3030, 0x303D, 0x3297, 0x3299}
    )


def demojize_fallback(text: str) -> str:
    pieces: list[str] = []
    for ch in text:
        if ch in {"\ufe0f", "\u200d"}:
            continue
        if ch in EMOJI_FALLBACK_NAMES:
            pieces.append(f" <{EMOJI_FALLBACK_NAMES[ch]}> ")
        elif is_probably_emoji(ch):
            name = unicodedata.name(ch, "").lower().replace(" ", "_").replace("-", "_")
            pieces.append(f" <{normalize_tag_name(name)}> " if name else " ")
        else:
            pieces.append(ch)
    return "".join(pieces)


def demojize_to_angle(text: str) -> str:
    if emoji_lib is not None:
        text = emoji_lib.demojize(text, delimiters=("<", ">"), language="en")
    else:
        text = demojize_fallback(text)
    return sanitize_demojized_tags(text)


def normalize_emoticons(text: str) -> str:
    replacements = [
        (r"(?i)(?:x|=|:)-?d+", " <laugh_emoticon> "),
        (r"(?:=|:|;)-?\)+", " <smile_emoticon> "),
        (r"(?:=|:|;)-?\(+", " <sad_emoticon> "),
        (r"(?i)t[_\-.]?t", " <cry_emoticon> "),
        (r"<3+", " <heart_emoticon> "),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text


def protect_heart_emoticons(text: str) -> str:
    return re.sub(r"<3+", f" {HEART_EMOTICON_PLACEHOLDER} ", text)


def restore_heart_emoticons(text: str) -> str:
    return text.replace(HEART_EMOTICON_PLACEHOLDER, " <heart_emoticon> ")


def remove_raw_angle_brackets(text: str) -> str:
    return text.replace("<", " ").replace(">", " ")


def remove_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def remove_post_id_hashtags(text: str, resources: PreprocessResources) -> str:
    for pattern in resources.hashtag_remove_regex:
        text = pattern.sub(" ", text)
    return text


def remove_or_keep_hashtags(text: str, resources: PreprocessResources) -> str:
    def replace_hashtag(match: re.Match[str]) -> str:
        hashtag = match.group(0)
        body = hashtag.lstrip("#")
        normalized = normalize_hashtag_key(body)
        if normalized in resources.hashtag_remove_normalized:
            return " "
        return " " + body.replace("_", " ") + " "

    return HASHTAG_RE.sub(replace_hashtag, text)


def reduce_repeated_letters(text: str) -> str:
    return REPEATED_ALPHA_RE.sub(r"\1", text)


def expand_teencode(text: str, resources: PreprocessResources) -> str:
    if resources.teencode_pattern is None:
        return text
    return resources.teencode_pattern.sub(
        lambda match: resources.teencode_map.get(match.group(1), match.group(1)),
        text,
    )


def read_two_digits(number: int) -> str:
    if number < 10:
        return DIGIT_WORDS[number]
    tens, ones = divmod(number, 10)
    if tens == 1:
        if ones == 0:
            return "mười"
        if ones == 5:
            return "mười lăm"
        return "mười " + DIGIT_WORDS[ones]

    parts = [DIGIT_WORDS[tens], "mươi"]
    if ones == 1:
        parts.append("mốt")
    elif ones == 4:
        parts.append("bốn")
    elif ones == 5:
        parts.append("lăm")
    elif ones:
        parts.append(DIGIT_WORDS[ones])
    return " ".join(parts)


def read_three_digits(number: int, force_hundreds: bool = False) -> str:
    if number == 0:
        return "không" if not force_hundreds else ""

    hundreds, rest = divmod(number, 100)
    parts: list[str] = []
    if hundreds:
        parts.extend([DIGIT_WORDS[hundreds], "trăm"])
    elif force_hundreds:
        parts.extend(["không", "trăm"])

    if rest:
        if parts and rest < 10:
            parts.append("lẻ")
            parts.append(DIGIT_WORDS[rest])
        else:
            parts.append(read_two_digits(rest))

    return " ".join(parts)


def read_int_vietnamese(number: int) -> str:
    if number == 0:
        return DIGIT_WORDS[0]
    if number < 0:
        return "âm " + read_int_vietnamese(abs(number))

    units = ["", "nghìn", "triệu", "tỷ"]
    groups: list[int] = []
    while number:
        groups.append(number % 1000)
        number //= 1000

    parts: list[str] = []
    highest_index = len(groups) - 1
    for index in range(highest_index, -1, -1):
        group = groups[index]
        if group == 0:
            continue
        force_hundreds = index < highest_index and group < 100
        group_text = read_three_digits(group, force_hundreds=force_hundreds)
        unit = units[index] if index < len(units) else ""
        parts.append((group_text + " " + unit).strip())

    return " ".join(parts)


def read_digit_sequence(number_text: str) -> str:
    return " ".join(DIGIT_WORDS[int(ch)] for ch in number_text if ch.isdigit())


def number_token_to_words(number_text: str) -> str:
    number_text = number_text.strip()
    if len(number_text) > 6 or (len(number_text) > 1 and number_text.startswith("0")):
        return read_digit_sequence(number_text)
    return read_int_vietnamese(int(number_text))


def normalize_unit_suffix(unit: str) -> str:
    return strip_accents(unit.lower())


def unit_suffix_to_words(unit: str) -> str:
    normalized = normalize_unit_suffix(unit)
    return UNIT_SUFFIX_MAP.get(normalized, unit.lower())


def alpha_index(index: int) -> str:
    letters: list[str] = []
    while True:
        index, remainder = divmod(index, 26)
        letters.append(chr(ord("a") + remainder))
        if index == 0:
            break
        index -= 1
    return "".join(reversed(letters))


def protect_k_amounts(text: str) -> tuple[str, dict[str, str]]:
    preserved: dict[str, str] = {}

    def replace_amount(match: re.Match[str]) -> str:
        placeholder = f"keepkamount{alpha_index(len(preserved))}token"
        preserved[placeholder] = match.group(0).lower()
        return f" {placeholder} "

    return K_AMOUNT_RE.sub(replace_amount, text), preserved


def restore_k_amounts(text: str, preserved: dict[str, str]) -> str:
    for placeholder, original in preserved.items():
        text = text.replace(placeholder, original)
    return text


def mixed_alnum_token_to_words(token: str) -> str:
    parts = re.findall(r"\d+|[a-zà-ỹđ]+", token.lower(), flags=re.IGNORECASE)
    return " ".join(number_token_to_words(part) if part.isdigit() else part for part in parts)


def convert_numbers_to_words(text: str) -> str:
    text, preserved_k_amounts = protect_k_amounts(text)
    text = PERCENT_RE.sub(
        lambda match: number_token_to_words(match.group(1)) + " phần trăm",
        text,
    )
    text = TIME_RE.sub(
        lambda match: number_token_to_words(match.group(1))
        + " giờ "
        + number_token_to_words(match.group(2)),
        text,
    )
    text = HEIGHT_RE.sub(
        lambda match: number_token_to_words(match.group(1))
        + " mét "
        + number_token_to_words(match.group(2)),
        text,
    )
    text = NUMBER_SUFFIX_RE.sub(
        lambda match: number_token_to_words(match.group(1)) + " " + match.group(2).lower(),
        text,
    )
    text = NUMBER_UNIT_RE.sub(
        lambda match: number_token_to_words(match.group(1))
        + " "
        + unit_suffix_to_words(match.group(2)),
        text,
    )
    text = LETTER_NUMBER_RE.sub(
        lambda match: match.group(1).lower() + " " + number_token_to_words(match.group(2)),
        text,
    )
    text = MIXED_ALNUM_RE.sub(lambda match: mixed_alnum_token_to_words(match.group(0)), text)
    text = DECIMAL_RE.sub(
        lambda match: number_token_to_words(match.group(1))
        + " phẩy "
        + read_digit_sequence(match.group(2)),
        text,
    )
    text = INTEGER_RE.sub(lambda match: number_token_to_words(match.group(0)), text)
    return restore_k_amounts(text, preserved_k_amounts)


def process_punctuation(text: str) -> str:
    text = re.sub(r"!+", " <exclamation> ", text)
    text = re.sub(r"\?+", " <question> ", text)
    text = re.sub(r"…+|\.{2,}", " ", text)
    text = re.sub(r"[.,;:\"'`“”‘’]+", " ", text)
    return text


def remove_non_emotional_special_chars(text: str) -> str:
    pieces: list[str] = []
    inside_tag = False
    for ch in text:
        if ch == "<":
            inside_tag = True
            pieces.append(ch)
        elif ch == ">":
            inside_tag = False
            pieces.append(ch)
        elif ch == "_" and inside_tag:
            pieces.append(ch)
        elif ch.isalnum() or ch.isspace():
            pieces.append(ch)
        else:
            pieces.append(" ")
    return "".join(pieces)


def replace_underscores_outside_tags(text: str) -> str:
    pieces: list[str] = []
    inside_tag = False
    for ch in text:
        if ch == "<":
            inside_tag = True
            pieces.append(ch)
        elif ch == ">":
            inside_tag = False
            pieces.append(ch)
        elif ch == "_" and not inside_tag:
            pieces.append(" ")
        else:
            pieces.append(ch)
    return "".join(pieces)


def preprocess_text(text: object, resources: PreprocessResources) -> str:
    text = "" if pd.isna(text) else normalize_unicode(str(text))
    text = text.lower()
    text = remove_urls(text)
    text = protect_heart_emoticons(text)
    text = remove_raw_angle_brackets(text)
    text = demojize_to_angle(text)
    text = text.lower()
    text = normalize_emoticons(text)
    text = restore_heart_emoticons(text)
    text = normalize_angle_sections(text)
    text = remove_post_id_hashtags(text, resources)
    text = remove_or_keep_hashtags(text, resources)
    text = reduce_repeated_letters(text)
    text = expand_teencode(text, resources)
    text = replace_underscores_outside_tags(text)
    text = convert_numbers_to_words(text)
    text = process_punctuation(text)
    text = remove_non_emotional_special_chars(text)
    return normalize_spaces(text)


def preprocess_dataframe(
    df: pd.DataFrame,
    resources: PreprocessResources,
    content_column: str = CONTENT_COLUMN,
    output_column: str = OUTPUT_COLUMN,
    show_progress: bool = True,
) -> pd.DataFrame:
    if output_column == content_column:
        raise ValueError(
            f"Output column must be different from raw content column '{content_column}'."
        )
    if content_column not in df.columns:
        raise ValueError(f"Column '{content_column}' not found. Available columns: {list(df.columns)}")

    result = df.copy()
    raw_content = df[content_column].copy(deep=True)
    texts = raw_content.fillna("").astype(str)
    iterator = texts
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(texts, total=len(texts), desc="Preprocessing")
        except ImportError:
            iterator = texts
    result[output_column] = [preprocess_text(text, resources) for text in iterator]
    if not result[content_column].equals(raw_content):
        raise AssertionError(f"Raw column '{content_column}' was modified during preprocessing.")
    return result


def run_preprocessing(
    input_csv: Path = DEFAULT_INPUT_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    teencode_path: Path = DEFAULT_TEENCODE_PATH,
    hashtag_map_path: Path = DEFAULT_HASHTAG_MAP_PATH,
    content_column: str = CONTENT_COLUMN,
    output_column: str = OUTPUT_COLUMN,
    show_progress: bool = True,
) -> pd.DataFrame:
    resources = load_resources(teencode_path=teencode_path, hashtag_map_path=hashtag_map_path)
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    result = preprocess_dataframe(
        df,
        resources=resources,
        content_column=content_column,
        output_column=output_column,
        show_progress=show_progress,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess merch/student sentiment dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--teencode", type=Path, default=DEFAULT_TEENCODE_PATH)
    parser.add_argument("--hashtag-map", type=Path, default=DEFAULT_HASHTAG_MAP_PATH)
    parser.add_argument("--content-column", default=CONTENT_COLUMN)
    parser.add_argument("--output-column", default=OUTPUT_COLUMN)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_preprocessing(
        input_csv=args.input,
        output_csv=args.output,
        teencode_path=args.teencode,
        hashtag_map_path=args.hashtag_map,
        content_column=args.content_column,
        output_column=args.output_column,
        show_progress=not args.no_progress,
    )
    empty_rows = int((result[args.output_column].fillna("").str.len() == 0).sum())
    raw_unchanged = pd.read_csv(args.input, encoding="utf-8-sig")[args.content_column].equals(
        result[args.content_column]
    )
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Rows: {len(result)}")
    print(f"Columns: {list(result.columns)}")
    print(f"Raw column unchanged: {raw_unchanged}")
    print(f"Empty preprocessed rows: {empty_rows}")
    print(f"emoji package available: {emoji_lib is not None}")


if __name__ == "__main__":
    main()
