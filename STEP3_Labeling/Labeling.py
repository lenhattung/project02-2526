# -*- coding: utf-8 -*-
"""
Bước gán nhãn cảm xúc tự động (bước 1, phần "Gán nhãn") - dùng PhoBERT
đã fine-tune sẵn trên UIT-VSFC (Vietnamese Students' Feedback Corpus).

Vì UIT-VSFC là bình luận SINH VIÊN VỀ GIẢNG VIÊN/MÔN HỌC, domain khá
gần với dữ liệu Facebook giáo dục của anh (so với các model sentiment
train trên review thương mại điện tử) -> khả năng cao cho kết quả tốt
hơn khi áp dụng trực tiếp (zero-shot, không cần train lại).

QUAN TRỌNG - CHƯA CHẠY THỬ THỰC TẾ:
Sandbox hiện tại không có quyền truy cập huggingface.co nên mình
KHÔNG THỂ tải và test model này. Anh cần chạy trên máy có mạng, và
đặc biệt LƯU Ý bước "Kiểm tra mapping nhãn" bên dưới - đây là lỗi rất
dễ gặp khi tích hợp model ngoài (mapping nhãn có thể ngược thứ tự).

Cài đặt: pip install transformers torch pandas tqdm underthesea --break-system-packages
"""

import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize
from tqdm import tqdm

tqdm.pandas()

MODEL_NAME = "tmt3103/VSFC-sentiment-classify-phoBERT"

# Nhãn đích theo đúng yêu cầu của anh
LABEL_TICH_CUC = 2
LABEL_TRUNG_LAP = 1
LABEL_TIEU_CUC = 0

# Từ khóa để nhận diện nhãn của MODEL (không phân biệt hoa/thường,
# có dấu/không dấu) và map sang nhãn đích 0/1/2 của anh. Cách làm này
# AN TOÀN HƠN nhiều so với việc giả định thứ tự index 0/1/2 của model
# trùng với thứ tự anh muốn - đây là lỗi tích hợp rất hay gặp.
LABEL_TEXT_TO_TARGET = {
    "positive": LABEL_TICH_CUC,
    "pos": LABEL_TICH_CUC,
    "tích cực": LABEL_TICH_CUC,
    "tich cuc": LABEL_TICH_CUC,
    "negative": LABEL_TIEU_CUC,
    "neg": LABEL_TIEU_CUC,
    "tiêu cực": LABEL_TIEU_CUC,
    "tieu cuc": LABEL_TIEU_CUC,
    "neutral": LABEL_TRUNG_LAP,
    "neu": LABEL_TRUNG_LAP,
    "trung lập": LABEL_TRUNG_LAP,
    "trung lap": LABEL_TRUNG_LAP,
}


def load_model():
    print(f"Đang tải model {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Đã tải model xong. Dùng device: {device}")

    # ==== BƯỚC QUAN TRỌNG NHẤT: kiểm tra mapping nhãn của model ====
    id2label = model.config.id2label
    print("\n>>> id2label CỦA MODEL (bắt buộc phải xem qua trước khi tin kết quả):")
    for idx, label in id2label.items():
        print(f"    index {idx} -> '{label}'")

    # Tự xây bảng map index của MODEL -> nhãn đích 0/1/2 của anh,
    # dựa trên NỘI DUNG CHỮ của label (không dựa vào thứ tự index)
    index_to_target = {}
    unmapped = []
    for idx, label in id2label.items():
        key = str(label).strip().lower()
        if key in LABEL_TEXT_TO_TARGET:
            index_to_target[int(idx)] = LABEL_TEXT_TO_TARGET[key]
        else:
            unmapped.append((idx, label))

    if unmapped:
        raise ValueError(
            f"Không map được các nhãn sau của model sang 0/1/2: {unmapped}. "
            f"Anh cần tự bổ sung vào LABEL_TEXT_TO_TARGET cho khớp với "
            f"nhãn thật của model (xem log id2label phía trên)."
        )

    print(">>> Bảng map index model -> nhãn đích (0=tiêu cực,1=trung lập,2=tích cực):")
    print("   ", index_to_target)
    print()

    return tokenizer, model, device, index_to_target


def preprocess_for_phobert(text: str) -> str:
    """
    PhoBERT được huấn luyện trên văn bản ĐÃ TÁCH TỪ (từ ghép nối bằng
    dấu gạch dưới, VD: "sinh_viên"). Cần áp dụng word_tokenize trước
    khi đưa vào model để khớp đúng định dạng huấn luyện gốc - bỏ qua
    bước này có thể làm giảm đáng kể độ chính xác.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text
    try:
        return word_tokenize(text, format="text")
    except Exception:
        return text


@torch.no_grad()
def predict_batch(texts, tokenizer, model, device, index_to_target, max_length=256):
    """Dự đoán nhãn cho 1 batch câu, trả về (list nhãn 0/1/2, list confidence)."""
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    ).to(device)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidences, pred_indices = probs.max(dim=-1)

    labels = [index_to_target[int(i)] for i in pred_indices.cpu().tolist()]
    confidences = confidences.cpu().tolist()
    return labels, confidences


def run(input_path: str, output_path: str, content_col: str = "content",
        batch_size: int = 32, audit_sample_size: int = 1000, audit_path: str = None):

    tokenizer, model, device, index_to_target = load_model()

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    print(f"Tổng số dòng: {len(df)}")

    # --- Tiền xử lý: tách từ theo chuẩn PhoBERT ---
    print("Đang tách từ (word segmentation) cho toàn bộ dữ liệu...")
    df["_segmented"] = df[content_col].progress_apply(preprocess_for_phobert)

    # --- Dự đoán theo batch ---
    print("Đang dự đoán nhãn cảm xúc theo batch...")
    all_labels = []
    all_confidences = []
    texts = df["_segmented"].fillna("").astype(str).tolist()

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        labels, confidences = predict_batch(batch, tokenizer, model, device, index_to_target)
        all_labels.extend(labels)
        all_confidences.extend(confidences)

    df["sentiment_label"] = all_labels
    df["sentiment_confidence"] = all_confidences
    df.drop(columns=["_segmented"], inplace=True)

    # --- Thống kê nhanh phân bố nhãn (để phát hiện bất thường sớm,
    # VD: model dự đoán gần như toàn 1 nhãn -> có khả năng mapping sai
    # hoặc domain mismatch nghiêm trọng) ---
    print("\nPhân bố nhãn dự đoán:")
    print(df["sentiment_label"].value_counts().rename({2: "Tích cực (2)", 1: "Trung lập (1)", 0: "Tiêu cực (0)"}))

    # --- Lưu kết quả ---
    if output_path.endswith(".csv"):
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(output_path, index=False)
    print(f"\nĐã lưu kết quả: {output_path}")

    # --- Xuất mẫu audit: ưu tiên lấy đều theo từng nhãn (stratified)
    # thay vì random thuần, để đảm bảo audit đủ số lượng ở cả 3 nhãn,
    # kể cả khi phân bố nhãn lệch (VD: tiêu cực chỉ chiếm 5% dữ liệu) ---
    if audit_path is None:
        audit_path = output_path.rsplit(".", 1)[0] + "_audit_sample.xlsx"

    n_per_label = max(1, audit_sample_size // 3)
    audit_parts = []
    for label_value in [0, 1, 2]:
        subset = df[df["sentiment_label"] == label_value]
        take = min(n_per_label, len(subset))
        audit_parts.append(subset.sample(n=take, random_state=42))
    audit_df = pd.concat(audit_parts)[[content_col, "sentiment_label", "sentiment_confidence"]].copy()
    audit_df["nhan_dung_khong"] = ""  # cột để tick Đúng/Sai thủ công
    audit_df["nhan_dung_neu_sai"] = ""  # cột ghi lại nhãn đúng nếu model sai
    audit_df.to_excel(audit_path, index=False)
    print(f"Đã xuất mẫu audit ({len(audit_df)} dòng, chia đều 3 nhãn): {audit_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-label sentiment bằng PhoBERT (UIT-VSFC)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--content_col", default="content")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--audit_size", type=int, default=1000)
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        content_col=args.content_col,
        batch_size=args.batch_size,
        audit_sample_size=args.audit_size,
    )