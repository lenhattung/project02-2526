# -*- coding: utf-8 -*-
"""
Pipeline ẩn danh tên người - PHIÊN BẢN 2 (Transformer-based)
------------------------------------------------------------------
Dùng model NlpHUST/ner-vietnamese-electra-base (kiến trúc ELECTRA,
transformer dựa ngữ cảnh) thay cho underthesea (CRF + POS-based) ở
bản v1 (anonymize_names.py).

LƯU Ý QUAN TRỌNG:
- Đây KHÔNG phải PhoBERT thuần - hiện không có checkpoint PhoBERT-NER
  "dùng ngay" (ready-to-use) được maintain phổ biến, ổn định trên
  HuggingFace Hub. NlpHUST/ner-vietnamese-electra-base là lựa chọn
  transformer-based phổ biến nhất có thể dùng NGAY không cần fine-tune.
  Nếu anh cần đúng PhoBERT, phải tự fine-tune trên VLSP2018/PhoNER-
  COVID19 (mình có thể viết code này riêng khi anh có dữ liệu).
- Script này CHƯA được chạy thử thực tế (sandbox không có quyền truy
  cập huggingface.co để tải model) - anh cần chạy trên máy có mạng
  đầy đủ và báo lại kết quả để mình tiếp tục điều chỉnh nếu cần.

Cài đặt: pip install transformers torch pandas tqdm --break-system-packages
"""

import re
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

tqdm.pandas()

MODEL_NAME = "NlpHUST/ner-vietnamese-electra-base"

# ==========================================================
# 1. Xử lý mention Facebook có cấu trúc (giống hệt bản v1)
# ==========================================================

FB_MENTION_PATTERN = re.compile(r"@\[\d+:\d+:([^\]]+)\]")
FB_PROFILE_LINK_PATTERN = re.compile(
    r"(https?://)?(www\.)?facebook\.com/[^\s]+", re.IGNORECASE
)


def mask_structured_mentions(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = FB_MENTION_PATTERN.sub("(TENNGUOI)", text)
    text = FB_PROFILE_LINK_PATTERN.sub("(TENNGUOI)", text)
    return text


# ==========================================================
# 2. Load model NER (transformer-based)
#    aggregation_strategy="simple" -> model tự gộp các sub-token
#    liên tiếp cùng nhãn thành 1 entity, trả về luôn start/end
#    theo VỊ TRÍ KÝ TỰ trong câu gốc -> không cần tự ghép span
#    thủ công như bản v1 (đỡ rủi ro lỗi spacing/merge).
# ==========================================================

print(f"Đang tải model {MODEL_NAME} ...")
ner_pipeline = pipeline(
    "token-classification",
    model=MODEL_NAME,
    aggregation_strategy="simple",
)
print("Tải model xong.")


def get_person_spans_from_model(text: str):
    """
    Trả về danh sách (start, end) các entity được model gắn nhãn PERSON.
    Dò nhãn theo kiểu chứa 'PER' (không phân biệt hoa/thường) để tương
    thích với các biến thể đặt tên nhãn khác nhau (PER, PERSON, B-PER...).
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    try:
        entities = ner_pipeline(text)
    except Exception:
        return []

    spans = []
    for ent in entities:
        label = str(ent.get("entity_group", "")).upper()
        if "PER" in label:
            spans.append((ent["start"], ent["end"]))
    return spans


# ==========================================================
# 3. Lớp fallback dựa trên danh xưng (thầy/cô/anh/chị/sinh viên...)
#    Viết lại THUẦN REGEX trên văn bản gốc (không phụ thuộc tagged
#    tuple của underthesea nữa) để dùng độc lập với bất kỳ model NER
#    nào. Logic kế thừa từ bản v1 sau khi đã audit qua nhiều case thực
#    tế: viết hoa chuẩn, viết thường hoàn toàn, tên bị tách/gộp token.
# ==========================================================

COMMON_VN_SURNAMES = {
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ",
    "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý", "Đinh", "Đoàn",
    "Trịnh", "Trương", "Lâm", "Mai", "Tô", "Tăng", "Chu", "Cao",
}
SURNAMES_LOWER = {s.lower() for s in COMMON_VN_SURNAMES}

HONORIFIC_WORDS = [
    "thầy", "cô", "anh", "chị", "em", "bạn", "ông", "bà",
    "chú", "dì", "cậu", "mợ", "bác", "sếp", "sư",
    "sinh viên", "học sinh", "giáo viên", "giảng viên",
]
# Sort dài -> ngắn để regex ưu tiên khớp cụm dài trước (VD "sinh viên"
# trước khi thử khớp "sinh")
HONORIFIC_WORDS.sort(key=len, reverse=True)
HONORIFIC_PATTERN = re.compile(
    r"(?:" + "|".join(re.escape(w) for w in HONORIFIC_WORDS) + r")\b",
    re.IGNORECASE,
)

FUNCTION_STOPWORDS = {
    "là", "và", "với", "của", "cho", "này", "đó", "rồi", "không", "có",
    "đã", "sẽ", "đang", "vẫn", "cũng", "chỉ", "mà", "nhưng", "nếu", "thì",
    "nên", "phải", "ơi", "à", "nhé", "nhỉ", "vậy", "sao", "đâu", "gì",
    "ai", "khi", "trong", "ngoài", "trên", "dưới", "giữa",
}

PLAIN_WORD_RE = re.compile(r"^[^\W\d_]+$", re.UNICODE)
VN_UPPER_START = r"[A-ZĐÂÊÔƠƯÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ]"
LEADING_CAPITALIZED_RUN = re.compile(
    rf"^{VN_UPPER_START}[\w\u00C0-\u1EF9]*(?:\s+{VN_UPPER_START}[\w\u00C0-\u1EF9]*){{0,3}}"
)


def get_person_spans_from_honorific(text: str):
    """
    Quét toàn bộ text tìm các cụm 'danh xưng + tên người', trả về danh
    sách (start, end) của PHẦN TÊN (không bao gồm danh xưng).
    Hoạt động thuần trên chuỗi ký tự, không phụ thuộc bất kỳ model nào.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []

    spans = []
    for m in HONORIFIC_PATTERN.finditer(text):
        after = text[m.end():]
        # Bỏ qua khoảng trắng ngay sau danh xưng
        stripped_len = len(after) - len(after.lstrip())
        offset = m.end() + stripped_len
        after = after.lstrip()
        if not after:
            continue

        # Ưu tiên 1: chuỗi viết hoa liên tiếp
        cap_match = LEADING_CAPITALIZED_RUN.match(after)
        if cap_match:
            start = offset
            end = offset + cap_match.end()
            spans.append((start, end))
            continue

        # Ưu tiên 2: tên viết thường, so khớp họ VN (không phân biệt
        # hoa/thường), dừng lại nhờ FUNCTION_STOPWORDS, tối đa 3 từ
        words_iter = list(re.finditer(r"[^\W\d_]+", after, re.UNICODE))
        if not words_iter:
            continue
        first_word = words_iter[0].group()
        if first_word.lower() not in SURNAMES_LOWER:
            continue

        collected_end = words_iter[0].end()
        count = 1
        for w in words_iter[1:3]:
            # Phải liền kề (chỉ cách bởi khoảng trắng) với từ trước đó
            gap = after[collected_end:w.start()]
            if gap.strip() != "":
                break
            if w.group().lower() in FUNCTION_STOPWORDS:
                break
            collected_end = w.end()
            count += 1

        if count >= 2:
            spans.append((offset, offset + collected_end))

    return spans


# ==========================================================
# 4. Ghép span từ model + fallback danh xưng, merge trùng lặp,
#    thay thế theo vị trí ký tự (giữ nguyên spacing/dấu câu gốc)
# ==========================================================

def anonymize_pipeline(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text

    masked = mask_structured_mentions(text)

    model_spans = get_person_spans_from_model(masked)
    honorific_spans = get_person_spans_from_honorific(masked)

    all_spans = model_spans + honorific_spans
    if not all_spans:
        return masked

    all_spans.sort(key=lambda s: s[0])
    merged = [all_spans[0]]
    for start, end in all_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    output_chunks = []
    cursor = 0
    for start, end in merged:
        output_chunks.append(masked[cursor:start])
        output_chunks.append("(TENNGUOI)")
        cursor = end
    output_chunks.append(masked[cursor:])
    return "".join(output_chunks)


# ==========================================================
# 5. Áp dụng lên toàn bộ dataset (giống cấu trúc bản v1)
# ==========================================================

def run(input_path: str, output_path: str, content_col: str = "content",
        min_len: int = 20, audit_sample_size: int = 200, audit_path: str = None):

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    print(f"Tổng số dòng ban đầu: {len(df)}")
    df = df[df[content_col].astype(str).str.len() > min_len].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"Số dòng sau khi lọc > {min_len} ký tự: {len(df)}")

    print("Đang chạy NER ẩn danh tên người (ELECTRA transformer)...")
    df["content_anonymized"] = df[content_col].progress_apply(anonymize_pipeline)

    if output_path.endswith(".csv"):
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(output_path, index=False)
    print(f"Đã lưu kết quả: {output_path}")

    if audit_path is None:
        audit_path = output_path.rsplit(".", 1)[0] + "_audit_sample.xlsx"
    sample_size = min(audit_sample_size, len(df))
    audit_df = df.sample(n=sample_size, random_state=42)[
        [content_col, "content_anonymized"]
    ].copy()
    audit_df["dung_khong"] = ""
    audit_df.to_excel(audit_path, index=False)
    print(f"Đã xuất mẫu audit ({sample_size} dòng): {audit_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ẩn danh tên người (v2 - ELECTRA transformer-based)"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--content_col", default="content")
    parser.add_argument("--min_len", type=int, default=20)
    parser.add_argument("--audit_size", type=int, default=200)
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        content_col=args.content_col,
        min_len=args.min_len,
        audit_sample_size=args.audit_size,
    )