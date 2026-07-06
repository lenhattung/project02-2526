# -*- coding: utf-8 -*-
"""
Pipeline ẩn danh tên người (Bước 1 - CRF-based, dùng underthesea)
------------------------------------------------------------------
Input : CSV/Excel có cột chứa nội dung văn bản (mặc định tên cột 'content')
Output: file mới thêm cột 'content_anonymized' (đã thay tên người bằng [NAME])
        + file audit (mẫu ngẫu nhiên) để kiểm tra thủ công Precision/Recall

Cài đặt: pip install underthesea pandas --break-system-packages
"""

import re
import random
import pandas as pd
from underthesea import ner
from tqdm import tqdm

tqdm.pandas()

# ==========================================================
# 1. Xử lý các trường hợp có cấu trúc (Facebook mention/tag)
#    trước khi đưa qua NER — vì đây là dữ liệu có pattern rõ ràng,
#    regex xử lý nhanh và chính xác 100%, không cần tốn NER cho phần này.
# ==========================================================

# Pattern mention kiểu cũ của FB export: @[100012345:0:Nguyễn Văn A]
FB_MENTION_PATTERN = re.compile(r"@\[\d+:\d+:([^\]]+)\]")

# Pattern link profile dạng facebook.com/ten.nguoi hoặc facebook.com/profile.php?id=...
FB_PROFILE_LINK_PATTERN = re.compile(
    r"(https?://)?(www\.)?facebook\.com/[^\s]+", re.IGNORECASE
)


def mask_structured_mentions(text: str):
    """Thay các mention/link profile FB bằng [NAME]/(LINK) trước khi NER."""
    if not isinstance(text, str):
        return text
    text = FB_MENTION_PATTERN.sub("[NAME]", text)
    text = FB_PROFILE_LINK_PATTERN.sub("[NAME]", text)
    return text


# ==========================================================
# 2. Chạy NER (underthesea) trên phần văn bản còn lại
#    để bắt tên gõ tay trong nội dung (không phải mention)
# ==========================================================

COMMON_VN_SURNAMES = {
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ",
    "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý", "Đinh", "Đoàn",
    "Trịnh", "Trương", "Lâm", "Mai", "Tô", "Tăng", "Chu", "Cao",
}

HONORIFIC_WORDS = {
    "thầy", "cô", "anh", "chị", "em", "bạn", "ông", "bà",
    "chú", "dì", "cậu", "mợ", "bác", "sếp", "sư", "mr", "mrs", "ms",
    "sinh viên", "học sinh", "giáo viên", "giảng viên",
}

# Các từ chức năng cực kỳ phổ biến, gần như không bao giờ là 1 phần của
# tên riêng -> dùng làm điểm DỪNG khi gom từ viết thường xuyên qua
# nhiều token (vì không có tín hiệu viết hoa để biết tên kết thúc ở đâu).
FUNCTION_STOPWORDS = {
    "là", "và", "với", "của", "cho", "này", "đó", "rồi", "không", "có",
    "đã", "sẽ", "đang", "vẫn", "cũng", "chỉ", "mà", "nhưng", "nếu", "thì",
    "nên", "phải", "ơi", "à", "nhé", "nhỉ", "vậy", "sao", "đâu", "gì",
    "ai", "khi", "trong", "ngoài", "trên", "dưới", "giữa",
}


VN_UPPER_START = r"[A-ZĐÂÊÔƠƯÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ]"
LEADING_CAPITALIZED_RUN = re.compile(
    rf"^{VN_UPPER_START}[\w\u00C0-\u1EF9]*(?:\s+{VN_UPPER_START}[\w\u00C0-\u1EF9]*){{0,3}}"
)


def extract_leading_capitalized_run(s: str):
    """
    Trích xuất chuỗi từ viết hoa liên tiếp ở ĐẦU chuỗi s (tối đa 4 từ).
    Trả về None nếu từ đầu tiên không viết hoa.

    Dùng để tách phần "tên người" ra khỏi 1 token bị underthesea gộp
    nhầm với từ phía sau (VD: "Trần Thị Bích khen" -> "Trần Thị Bích"),
    KHÔNG phụ thuộc vào POS/NER tag của token đó (vì đã thấy các tag
    này không đáng tin trong nhiều trường hợp).
    """
    match = LEADING_CAPITALIZED_RUN.match(s)
    return match.group(0) if match else None


SURNAMES_LOWER = {s.lower() for s in COMMON_VN_SURNAMES}
PLAIN_WORD_RE = re.compile(r"^[^\W\d_]+$", re.UNICODE)  # chỉ gồm chữ cái, không số/ký tự đặc biệt


def collect_plain_words_after(tagged, start_idx: int, max_words: int = 4):
    """
    Gom tối đa max_words từ 'thường' (chỉ chữ cái, không số/ký tự đặc biệt)
    bắt đầu từ token start_idx, có thể XUYÊN QUA NHIỀU TOKEN (vì
    underthesea đôi khi tách tên thành nhiều token nhỏ, VD: "phạm" tách
    riêng khỏi "thế an" phía sau). Dừng lại khi:
      - gặp từ không phải chữ cái thuần (số, dấu câu...), hoặc
      - gặp từ nằm trong FUNCTION_STOPWORDS (là, và, ơi, không...) —
        dấu hiệu đã ra khỏi phạm vi tên riêng, hoặc
      - đã đủ max_words từ.
    """
    n = len(tagged)
    words = []
    idx = start_idx
    while idx < n and len(words) < max_words:
        token_str = tagged[idx][0]
        subwords = token_str.split(" ")
        hit_stop = False
        for sw in subwords:
            if not PLAIN_WORD_RE.match(sw) or sw.lower() in FUNCTION_STOPWORDS:
                hit_stop = True
                break
            words.append(sw)
            if len(words) >= max_words:
                hit_stop = True
                break
        if hit_stop:
            break
        idx += 1
    return words


def extract_name_case_insensitive(words):
    """
    Nhận vào danh sách từ (đã gom bằng collect_plain_words_after), kiểm
    tra từ ĐẦU TIÊN có khớp 1 họ Việt Nam phổ biến không (không phân
    biệt hoa/thường). Yêu cầu tối thiểu 2 từ để tăng độ tin cậy (tránh
    trường hợp chỉ trùng họ ngẫu nhiên với 1 từ thường bất kỳ).
    """
    if len(words) < 2:
        return None
    if words[0].lower() not in SURNAMES_LOWER:
        return None
    return " ".join(words)


def honorific_based_per_detection(tagged, extend_to_next_token: bool = True):
    """
    Bổ sung phát hiện PER dựa trên DANH XƯNG (thầy/cô/anh/chị/bạn/sinh
    viên...), độc lập với nhãn NER/POS trả về.

    Lý do cần lớp này: underthesea đôi khi gắn NHẦM HẲN sang nhãn khác
    (VD: "thầy Lê Nhật Tùng" bị gắn LOC thay vì PER), gộp nhầm cả cụm
    tên + từ phía sau thành 1 token (VD: "Trần Thị Bích khen"), hoặc
    ngược lại TÁCH RỜI 1 tên thành nhiều token nhỏ (VD: "phạm" tách
    khỏi "thế an"). Vì vậy lớp này chỉ dựa vào: (1) từ danh xưng đứng
    trước, (2) chuỗi từ liên tiếp ngay sau đó (có thể xuyên nhiều
    token) — không tin vào POS/NER của các token này.

    2 nhánh phát hiện:
      - Viết hoa chuẩn: dùng chuỗi viết hoa liên tiếp làm ranh giới tên
        (đáng tin cậy nhất vì viết hoa là tín hiệu rõ ràng).
      - Viết thường hoàn toàn: so khớp họ VN không phân biệt hoa/thường,
        dừng lại đúng chỗ nhờ FUNCTION_STOPWORDS (là, và, ơi...) để
        không nuốt nhầm sang phần câu tiếp theo.
    """
    extra_phrases = []
    n = len(tagged)
    for i, item in enumerate(tagged):
        word = item[0]
        if word.lower() in HONORIFIC_WORDS and i + 1 < n:
            # Ưu tiên 1: chuỗi viết hoa liên tiếp trong token ngay sau
            # (đã tự nhiên dừng đúng chỗ nhờ tín hiệu viết hoa, kể cả
            # khi token đó bị gộp nhầm với từ phía sau)
            next_word = tagged[i + 1][0]
            leading = extract_leading_capitalized_run(next_word)
            if leading:
                extra_phrases.append(leading)
                continue

            # Ưu tiên 2: tên viết thường hoàn toàn, gom xuyên nhiều
            # token, dừng lại nhờ FUNCTION_STOPWORDS.
            # Giới hạn 3 từ (Họ + đệm + tên) thay vì 4, vì đây là độ dài
            # phổ biến nhất của tên Việt Nam — nếu để 4, dễ nuốt nhầm
            # từ đầu câu tiếp theo khi tên chỉ có 3 từ (VD: "nguyễn văn
            # nam nghỉ học" -> "nghỉ" bị nuốt nhầm nếu cap = 4).
            collected = collect_plain_words_after(tagged, i + 1, max_words=3)
            leading_ci = extract_name_case_insensitive(collected)
            if leading_ci:
                extra_phrases.append(leading_ci)
    return extra_phrases


def anonymize_person_names(text: str, extend_to_next_token: bool = True) -> str:
    """
    Nhận input là 1 câu/content, trả về câu đã thay các cụm từ
    được gắn nhãn B-PER/I-PER thành [NAME].

    Dùng cách thay thế theo VỊ TRÍ KÝ TỰ trong text gốc (không ghép lại
    từ token) để giữ nguyên dấu câu, khoảng trắng, xuống dòng... của
    văn bản gốc — tránh lỗi kiểu "dở quá , không như" (thừa khoảng trắng
    trước dấu phẩy) khi rebuild câu từ danh sách token.

    extend_to_next_token:
        True  -> bắt thêm phần tên bị "nuốt" vào từ ghép kế tiếp do lỗi
                 tách từ của underthesea (VD: "Hùng Minh môn" -> "Minh"
                 bị gộp nhầm với "môn"). Tăng RECALL nhưng có rủi ro
                 FALSE POSITIVE khi từ viết hoa kế tiếp không phải tên
                 (VD: "Nguyễn Văn A Đẹp trai quá" -> "Đẹp" có thể bị
                 nuốt nhầm do viết hoa tùy tiện, khá phổ biến trên FB).
        False -> chỉ dùng đúng span mà NER trả về, an toàn hơn về
                 Precision nhưng bỏ sót nhiều tên hơn (Recall thấp hơn).
        Khuyến nghị: bật True nếu ưu tiên không để lọt tên (bảo vệ
        privacy là ưu tiên hàng đầu); tắt False nếu ưu tiên giữ nguyên
        văn bản tối đa cho bước NLP tiếp theo.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text

    try:
        tagged = ner(text)
    except Exception:
        # Nếu NER lỗi (câu quá dài, ký tự lạ...) thì trả nguyên văn bản gốc
        # và log lại để kiểm tra riêng, KHÔNG âm thầm bỏ qua.
        return text

    # Xác định các đoạn từ liên tiếp là PER (gộp B-PER + I-PER kế tiếp)
    #
    # Xử lý thêm 1 lỗi phổ biến của underthesea: bộ tách từ (word
    # segmenter) đôi khi NHẦM GỘP phần còn lại của tên người với từ
    # tiếp theo thành 1 "từ ghép" sai (VD: "Hùng Minh môn hiểu biết"
    # bị tách thành token "Hùng" | "Minh môn" | "hiểu biết" — "Minh"
    # bị nuốt chung với "môn"). Vì vậy sau khi thấy 1 token gắn nhãn
    # PER, ta kiểm tra token liền sau: nếu nó là cụm nhiều từ và từ
    # ĐẦU của cụm đó viết hoa (dấu hiệu vẫn là tên riêng), thì coi
    # phần chữ hoa đầu đó cũng là 1 phần của tên và thay thế luôn.
    per_phrases = []
    n = len(tagged)
    for i, item in enumerate(tagged):
        word = item[0]
        ner_tag = item[-1]
        if ner_tag in ("B-PER", "I-PER"):
            per_phrases.append(word)

            # --- Kiểm tra token liền sau có bị nuốt tên không ---
            if extend_to_next_token and i + 1 < n:
                next_word = tagged[i + 1][0]
                next_tag = tagged[i + 1][-1]
                if next_tag == "O" and " " in next_word:
                    sub_words = next_word.split(" ", 1)
                    leading = sub_words[0]
                    # Từ đầu viết hoa (chữ cái đầu là hoa) -> khả năng
                    # cao vẫn là 1 phần tên riêng bị nuốt vào từ ghép
                    if leading and leading[0].isupper():
                        per_phrases.append(leading)

    # --- Bổ sung phát hiện dựa trên danh xưng (độc lập với nhãn NER) ---
    per_phrases.extend(
        honorific_based_per_detection(tagged, extend_to_next_token=extend_to_next_token)
    )

    if not per_phrases:
        return text

    result = text
    # Tìm vị trí ký tự của TỪNG phrase một cách ĐỘC LẬP (không dùng cursor
    # chung theo thứ tự trong per_phrases, vì per_phrases có thể đến từ
    # 2 nguồn khác nhau — NER-based và honorific-based — nên thứ tự
    # trong list chưa chắc khớp thứ tự xuất hiện thật trong câu).
    # Sau đó sort lại theo vị trí xuất hiện (start index) trước khi merge.
    raw_spans = []
    for phrase in per_phrases:
        idx = result.find(phrase)
        if idx == -1:
            continue
        raw_spans.append((idx, idx + len(phrase)))

    if not raw_spans:
        return text

    # Sort theo start, rồi dedupe/merge các span trùng hoặc lồng nhau
    raw_spans.sort(key=lambda s: s[0])
    spans = [raw_spans[0]]
    for start, end in raw_spans[1:]:
        prev_start, prev_end = spans[-1]
        if start <= prev_end:  # trùng hoặc lồng nhau -> merge
            spans[-1] = (prev_start, max(prev_end, end))
        else:
            spans.append((start, end))

    # Gộp các span LIỀN KỀ nhau (chỉ cách nhau bởi khoảng trắng) thành 1
    # span duy nhất -> tránh bị "[NAME] [NAME]" lặp lại khi 1 tên
    # bị chia thành 2 token do lỗi tách từ của underthesea.
    merged_spans = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged_spans[-1]
        gap = result[prev_end:start]
        if gap.strip() == "":  # giữa 2 span chỉ có khoảng trắng
            merged_spans[-1] = (prev_start, end)
        else:
            merged_spans.append((start, end))

    output_chunks = []
    cursor = 0
    for start, end in merged_spans:
        output_chunks.append(result[cursor:start])
        output_chunks.append("[NAME]")
        cursor = end
    output_chunks.append(result[cursor:])
    return "".join(output_chunks)


def anonymize_pipeline(text: str, extend_to_next_token: bool = True) -> str:
    """Full pipeline cho 1 dòng: mask mention có cấu trúc -> NER -> fallback dictionary."""
    text = mask_structured_mentions(text)
    text = anonymize_person_names(text, extend_to_next_token=extend_to_next_token)
    text = fallback_dictionary_name_at_start(text)
    return text


# ==========================================================
# 2b. Lớp fallback dựa trên từ điển họ Việt Nam phổ biến
#     Bổ sung cho NER — vì NER của underthesea phụ thuộc vào POS tag,
#     và POS hay nhầm "Họ Tên Đệm" thành danh từ thường khi đứng ở đầu
#     câu, đi liền trước 1 từ khác (VD: "Nguyễn Tuấn Kiệt trường..."
#     -> bị hiểu lầm là cụm "Kiệt trường"). Lỗi này đặc biệt phổ biến
#     trong caption/comment mạng xã hội vì câu hay mở đầu ngay bằng tên
#     người được nhắc đến (không có "Anh/Chị/Bạn" đứng trước để tạo
#     ngữ cảnh cho POS tagger).
#
# Cách làm: nếu câu (sau khi NER xử lý) VẪN bắt đầu bằng 1 cụm
# "Họ + (đệm) + tên" viết hoa đúng chuẩn và Họ nằm trong danh sách họ
# phổ biến VN, thì coi đó là PER và thay thế — CHỈ áp dụng ở đầu câu
# để hạn chế bắt nhầm (giảm rủi ro precision).
# ==========================================================

# Cụm 2-4 từ viết hoa liên tiếp ngay đầu chuỗi, từ đầu tiên là họ phổ biến
NAME_AT_START_PATTERN = re.compile(
    r"^(?:" + "|".join(COMMON_VN_SURNAMES) + r")"
    r"(?:\s+[A-ZĐÂÊÔƠƯÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ][\w\u00C0-\u1EF9]*){1,3}"
)


def fallback_dictionary_name_at_start(text: str) -> str:
    """
    Bắt bổ sung cụm tên người ở ĐẦU câu mà NER có thể đã bỏ sót.
    Chỉ áp dụng ở đầu chuỗi để hạn chế false positive.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text
    if text.strip().startswith("[NAME]"):
        # NER đã xử lý phần đầu rồi, không cần fallback nữa
        return text
    match = NAME_AT_START_PATTERN.match(text)
    if match:
        return "[NAME]" + text[match.end():]
    return text


# ==========================================================
# 3. Áp dụng lên toàn bộ dataset
# ==========================================================

def run(input_path: str, output_path: str, content_col: str = "content",
        min_len: int = 20, audit_sample_size: int = 200, audit_path: str = None,
        extend_to_next_token: bool = True):

    # --- Đọc dữ liệu ---
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    print(f"Tổng số dòng ban đầu: {len(df)}")

    # --- Lọc content > min_len ký tự (đúng bước 1 trong pipeline anh mô tả) ---
    df = df[df[content_col].astype(str).str.len() > min_len].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"Số dòng sau khi lọc > {min_len} ký tự: {len(df)}")

    # --- Chạy pipeline ẩn danh ---
    print("Đang chạy NER ẩn danh tên người (underthesea)...")
    df["content_anonymized"] = df[content_col].progress_apply(
        lambda t: anonymize_pipeline(t, extend_to_next_token=extend_to_next_token)
    )

    # --- Lưu kết quả ---
    if output_path.endswith(".csv"):
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(output_path, index=False)
    print(f"Đã lưu kết quả: {output_path}")

    # --- Xuất file audit: mẫu ngẫu nhiên để kiểm tra thủ công ---
    if audit_path is None:
        audit_path = output_path.rsplit(".", 1)[0] + "_audit_sample.xlsx"

    sample_size = min(audit_sample_size, len(df))
    audit_df = df.sample(n=sample_size, random_state=42)[
        [content_col, "content_anonymized"]
    ].copy()
    audit_df["dung_khong"] = ""  # cột để người kiểm tra tick Đ/S thủ công
    audit_df.to_excel(audit_path, index=False)
    print(f"Đã xuất mẫu audit ({sample_size} dòng): {audit_path}")

    return df


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Ẩn danh tên người trong content (CRF - underthesea)")
#     parser.add_argument("--input", required=True, help="Đường dẫn file input (csv/xlsx)")
#     parser.add_argument("--output", required=True, help="Đường dẫn file output (csv/xlsx)")
#     parser.add_argument("--content_col", default="content", help="Tên cột chứa nội dung")
#     parser.add_argument("--min_len", type=int, default=20, help="Độ dài tối thiểu để giữ lại")
#     parser.add_argument("--audit_size", type=int, default=200, help="Số dòng mẫu để audit thủ công")
#     parser.add_argument(
#         "--no_extend", action="store_true",
#         help="Tắt fallback mở rộng sang token kế tiếp (ưu tiên Precision, giảm Recall)"
#     )
#     args = parser.parse_args()

#     run(
#         input_path=args.input,
#         output_path=args.output,
#         content_col=args.content_col,
#         min_len=args.min_len,
#         audit_sample_size=args.audit_size,
#         extend_to_next_token=not args.no_extend,
#     )

print(anonymize_pipeline("Nguyễn Tuấn Kiệt trường hok có đọc đâu bà ơi 🥲🥲🥲"))

print(anonymize_pipeline("Hùng Minh môn hiểu biết về DNTU:)))???"))

print(anonymize_pipeline("Hoàng Nam bào nào mục xương thì thôi"))

print(anonymize_pipeline("Các bạn có biết thầy Lê Nhật Tùng không?"))

print(anonymize_pipeline("tui không biết cô nguyễn thị liệu"))

print(anonymize_pipeline("sinh viên phạm thế an là ai?"))

print(anonymize_pipeline("thông báo về vi phạm nguyên tắc cộng đồng"))