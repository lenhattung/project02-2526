"""
Preprocessing Pipeline cho DNTUConfession Sentiment Analysis
Quy trình 11 bước - mỗi bước là 1 function độc lập
"""

import re
import csv
import sqlite3
import os
from pathlib import Path

# ==========================================
# LOAD TỪ ĐIỂN
# ==========================================

DICT_DIR = Path("dictionaries")

def load_emoji_dict():
    """Load từ điển emoji từ CSV"""
    emoji_dict = {}
    with open(DICT_DIR / "emoji_dict.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emoji_dict[row["emoji"]] = row["text_replacement"]
    return emoji_dict

def load_teencode_dict():
    """Load từ điển teencode từ CSV (chỉ nhóm an_toàn)"""
    teencode_dict = {}
    with open(DICT_DIR / "teencode_dict.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["group"] == "an_toàn":
                teencode_dict[row["teencode"]] = row["standard"]
    return teencode_dict

def load_facebook_ui_patterns():
    """Load các pattern Facebook UI từ CSV"""
    patterns = []
    with open(DICT_DIR / "facebook_ui_patterns.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patterns.append((row["pattern"], row["type"]))
    return patterns

def load_lengthened_patterns():
    """Load pattern từ kéo dài từ CSV"""
    patterns = []
    with open(DICT_DIR / "lengthened_words.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patterns.append((row["pattern"], row["replacement"]))
    return patterns

def load_invalid_rules():
    """Load rules lọc bài không hợp lệ từ CSV"""
    rules = {}
    with open(DICT_DIR / "invalid_patterns.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rules[row["rule"]] = row
    return rules

# ==========================================
# 11 BƯỚC TIỀN XỬ LÝ
# ==========================================

def step1_remove_facebook_ui(text: str, patterns: list) -> str:
    """
    Bước 1: Xóa Facebook UI text
    (reaction count, timestamp, tên page, nút bấm)
    """
    for pattern, ptype in patterns:
        if ptype in ["ui_text", "timestamp", "reaction_count", "page_name"]:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text

def step2_remove_urls(text: str, patterns: list) -> str:
    """
    Bước 2: Xóa URL và link
    """
    for pattern, ptype in patterns:
        if ptype == "url":
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text

def step3_remove_phone_numbers(text: str, patterns: list) -> str:
    """
    Bước 3: Xóa số điện thoại (thông tin cá nhân)
    """
    for pattern, ptype in patterns:
        if ptype == "phone_number":
            text = re.sub(pattern, " ", text)
    return text

def step4_remove_hashtags(text: str, patterns: list) -> str:
    """
    Bước 4: Xóa hashtag confession (#Cfs2246_DNTU)
    Giữ lại nội dung, chỉ xóa mã hashtag
    """
    for pattern, ptype in patterns:
        if ptype == "hashtag_confession":
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    # Xóa các hashtag chung còn lại
    text = re.sub(r"#\w+", " ", text)
    return text

def step5_anonymize_names(text: str) -> str:
    """
    Bước 5: Ẩn danh tên người → [TÊN]
    Dùng NER của underthesea nếu có, fallback sang regex
    """
    try:
        from underthesea import ner
        entities = ner(text)
        for token, _, _, ner_tag in entities:
            if ner_tag in ["B-PER", "I-PER"]:
                text = text.replace(token, "[TÊN]")
    except ImportError:
        # Fallback: regex họ Việt Nam phổ biến
        ho_viet = (
            r"\b(Nguyễn|Trần|Lê|Phạm|Hoàng|Huỳnh|Phan|Vũ|Võ|"
            r"Đặng|Bùi|Đỗ|Hồ|Ngô|Dương|Lý)\s+"
            r"[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘ"
            r"ơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]"
            r"[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộ"
            r"ơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]+"
            r"(?:\s+[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘ"
            r"ơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]"
            r"[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộ"
            r"ơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]+)*"
        )
        text = re.sub(ho_viet, "[TÊN]", text)

    # Chuẩn hóa [TÊN] lặp liên tiếp
    text = re.sub(r"(\[TÊN\]\s*){2,}", "[TÊN] ", text)
    return text

def step6_convert_emoji(text: str, emoji_dict: dict) -> str:
    """
    Bước 6: Chuyển emoji → text cảm xúc
    Giữ ngữ cảnh sarcasm thông qua text thay thế
    """
    for emoji, replacement in emoji_dict.items():
        text = text.replace(emoji, f" {replacement} ")
    return text

def step7_normalize_lengthened(text: str, patterns: list) -> str:
    """
    Bước 7: Chuẩn hóa từ kéo dài
    buồnnnnn → buồnn (giữ 2 ký tự để giữ sắc thái nhấn mạnh)
    """
    for pattern, replacement in patterns:
        try:
            text = re.sub(pattern, replacement, text)
        except re.error:
            pass
    return text

def step8_normalize_teencode(text: str, teencode_dict: dict) -> str:
    """
    Bước 8: Chuẩn hóa teencode (chỉ nhóm an_toàn)
    Dùng word boundary để tránh thay sai ngữ cảnh
    """
    for teencode, standard in teencode_dict.items():
        pattern = r"(?<!\w)" + re.escape(teencode) + r"(?!\w)"
        text = re.sub(pattern, standard, text, flags=re.IGNORECASE)
    return text

def step9_normalize_whitespace(text: str) -> str:
    """
    Bước 9: Chuẩn hóa khoảng trắng và xuống dòng
    """
    text = re.sub(r"\n{3,}", "\n\n", text)   # Tối đa 2 dòng trống
    text = re.sub(r" {2,}", " ", text)         # Xóa space thừa
    text = re.sub(r"\t", " ", text)            # Tab → space
    text = text.strip()
    return text

def step10_filter_invalid(text: str, rules: dict) -> tuple[bool, str]:
    """
    Bước 10: Lọc bài không hợp lệ
    Trả về (is_valid, reason)
    """
    # Rule 1: Độ dài tối thiểu
    min_len = int(rules.get("min_length", {}).get("threshold", 20))
    if len(text) < min_len:
        return False, f"quá ngắn ({len(text)} ký tự)"

    # Rule 2: Tỷ lệ ký tự đặc biệt
    max_ratio = float(rules.get("max_special_ratio", {}).get("threshold", 0.5))
    special_chars = len(re.findall(r"[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệ"
                                   r"íìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
                                   r"A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊ"
                                   r"ÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ0-9\s]",
                                   text))
    if len(text) > 0 and special_chars / len(text) > max_ratio:
        return False, f"quá nhiều ký tự đặc biệt ({special_chars}/{len(text)})"

    # Rule 3: Chỉ còn [TÊN]
    text_no_tag = re.sub(r"\[TÊN\]", "", text).strip()
    if len(text_no_tag) < min_len:
        return False, "chỉ còn [TÊN] không có nội dung"

    # Rule 4: Toàn số
    if re.match(r"^[\d\s\.,]+$", text):
        return False, "chỉ toàn số"

    # Rule 5: Toàn dấu câu
    if re.match(r"^[\s\W]+$", text):
        return False, "chỉ toàn dấu câu"

    return True, "hợp lệ"

def step11_word_segmentation(text: str) -> str:
    """
    Bước 11: Tách từ tiếng Việt (bắt buộc cho PhoBERT)
    hạnh phúc → hạnh_phúc
    """
    try:
        from underthesea import word_tokenize
        text = word_tokenize(text, format="text")
    except ImportError:
        print("⚠️  underthesea chưa cài → bỏ qua bước tách từ")
        print("   Cài bằng: pip install underthesea")
    return text

# ==========================================
# PIPELINE TỔNG HỢP
# ==========================================

def preprocess(text: str,
               emoji_dict: dict,
               teencode_dict: dict,
               fb_patterns: list,
               lengthened_patterns: list,
               invalid_rules: dict) -> tuple[str, bool, str]:
    """
    Chạy toàn bộ 11 bước tiền xử lý
    Trả về (text_clean, is_valid, reason)
    """
    text = step1_remove_facebook_ui(text, fb_patterns)
    text = step2_remove_urls(text, fb_patterns)
    text = step3_remove_phone_numbers(text, fb_patterns)
    text = step4_remove_hashtags(text, fb_patterns)
    text = step5_anonymize_names(text)
    text = step6_convert_emoji(text, emoji_dict)
    text = step7_normalize_lengthened(text, lengthened_patterns)
    text = step8_normalize_teencode(text, teencode_dict)
    text = step9_normalize_whitespace(text)

    is_valid, reason = step10_filter_invalid(text, invalid_rules)
    if not is_valid:
        return text, False, reason

    text = step11_word_segmentation(text)

    return text, True, "hợp lệ"

# ==========================================
# ÁP DỤNG VÀO DATABASE
# ==========================================

def preprocess_database(db_path: str = "data/posts.db"):
    """Tiền xử lý toàn bộ bài viết trong database"""

    print("📚 Đang load từ điển...")
    emoji_dict         = load_emoji_dict()
    teencode_dict      = load_teencode_dict()
    fb_patterns        = load_facebook_ui_patterns()
    lengthened_patterns = load_lengthened_patterns()
    invalid_rules      = load_invalid_rules()
    print(f"   ✅ Emoji: {len(emoji_dict)} mục")
    print(f"   ✅ Teencode: {len(teencode_dict)} mục")
    print(f"   ✅ FB patterns: {len(fb_patterns)} mục")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Thêm cột mới nếu chưa có
    for col in ["content_clean", "is_valid", "invalid_reason", "label"]:
        try:
            cursor.execute(f"ALTER TABLE posts ADD COLUMN {col} TEXT")
            conn.commit()
        except:
            pass

    cursor.execute("SELECT id, content FROM posts WHERE content_clean IS NULL")
    posts = cursor.fetchall()
    print(f"\n📄 Xử lý {len(posts)} bài viết...\n")

    valid_count = 0
    invalid_count = 0

    for post_id, content in posts:
        if not content:
            continue

        clean, is_valid, reason = preprocess(
            content,
            emoji_dict,
            teencode_dict,
            fb_patterns,
            lengthened_patterns,
            invalid_rules
        )

        cursor.execute("""
            UPDATE posts
            SET content_clean = ?,
                is_valid = ?,
                invalid_reason = ?
            WHERE id = ?
        """, (clean, int(is_valid), reason, post_id))

        if is_valid:
            valid_count += 1
            print(f"✅ [{post_id}] {clean[:60]}...")
        else:
            invalid_count += 1
            print(f"❌ [{post_id}] Loại: {reason}")

    conn.commit()
    conn.close()

    print(f"\n{'='*50}")
    print(f"✅ Hợp lệ  : {valid_count} bài")
    print(f"❌ Loại bỏ : {invalid_count} bài")
    print(f"📁 Đã lưu vào cột content_clean trong {db_path}")

if __name__ == "__main__":
    preprocess_database()
