# -*- coding: utf-8 -*-
"""
BGE-M3 (BAAI/bge-m3) cho bài toán phân loại cảm xúc 3 lớp.

KHÁC VỚI 2 SCRIPT KIA (train_phobert.py, train_simcse.py): dùng cách
FEATURE EXTRACTION thay vì fine-tune end-to-end toàn bộ model:
  1. Đóng băng BGE-M3 (không train), chỉ dùng để trích embedding câu
  2. Train 1 classifier NHẸ (Logistic Regression / MLP nhỏ) lên trên
     các embedding đó

Lý do chọn hướng này thay vì fine-tune full:
  - BGE-M3 có ~568M tham số, nặng hơn nhiều so với PhoBERT/SimCSE-base
    (~135M) -> fine-tune full trên máy không có GPU rời (VD: LG Gram)
    sẽ rất chậm hoặc không khả thi.
  - BGE-M3 được thiết kế chính cho retrieval/embedding, không có sẵn
    classification head -> feature extraction là cách dùng tự nhiên,
    ít rủi ro hơn so với tự thêm head rồi fine-tune full 1 model lớn
    với dataset có thể không quá lớn.

LƯU Ý QUAN TRỌNG:
  - BGE-M3 dùng tokenizer đa ngôn ngữ (XLM-RoBERTa subword), KHÔNG cần
    tách từ tiếng Việt kiểu PhoBERT (underthesea/VnCoreNLP) - khác với
    2 script kia. Đưa thẳng câu tiếng Việt bình thường vào là được.
  - Cách trích dense embedding: lấy token CLS (vị trí đầu tiên của
    last_hidden_state) rồi chuẩn hóa L2 - đây là cách chính thức được
    xác nhận bởi team BAAI (không phải suy đoán).
  - Cùng random_state/tỷ lệ split với 2 script kia để đảm bảo cả 3
    model được đánh giá trên ĐÚNG CÙNG 1 tập test.

QUAN TRỌNG: Chưa chạy thử thực tế (sandbox không có quyền truy cập
huggingface.co). Anh chạy trên máy có mạng.

Cài đặt:
pip install transformers torch scikit-learn pandas --break-system-packages
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL_NAME = "BAAI/bge-m3"

ID2LABEL = {0: "tieu_cuc", 1: "trung_lap", 2: "tich_cuc"}

# GIỐNG HỆT train_phobert.py / train_simcse.py - bắt buộc để so sánh công bằng
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


def load_and_split(input_path: str, content_col: str, label_col: str = "sentiment_label"):
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    df = df[[content_col, label_col]].dropna().copy()
    df[label_col] = df[label_col].astype(int)

    train_df, temp_df = train_test_split(
        df, test_size=(TEST_SIZE + VAL_SIZE), random_state=RANDOM_STATE,
        stratify=df[label_col],
    )
    relative_test_size = TEST_SIZE / (TEST_SIZE + VAL_SIZE)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_size, random_state=RANDOM_STATE,
        stratify=temp_df[label_col],
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


@torch.no_grad()
def extract_embeddings(texts, tokenizer, model, device, batch_size: int = 16, max_length: int = 512):
    """
    Trích dense embedding cho danh sách câu, theo batch.
    Cách trích: CLS token (last_hidden_state[:, 0]) + L2 normalize -
    xác nhận chính thức từ team BAAI (không phải suy đoán/heuristic).
    """
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Trích embedding"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0]
        cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
        all_embeddings.append(cls_embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def run(input_path: str, output_dir: str, content_col: str = "content",
        label_col: str = "sentiment_label", classifier: str = "logreg",
        batch_size: int = 16):

    import os
    os.makedirs(output_dir, exist_ok=True)

    train_df, val_df, test_df = load_and_split(input_path, content_col, label_col)

    print(f"Đang tải model {MODEL_NAME} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device)
    print(f"Đã tải xong. Dùng device: {device}")

    # --- Trích embedding cho cả 3 tập ---
    # LƯU Ý: không tách từ (word segmentation) ở đây - khác với PhoBERT/
    # SimCSE - vì BGE-M3 dùng tokenizer subword đa ngôn ngữ, tự xử lý
    # được văn bản tiếng Việt thô.
    print("Trích embedding cho tập train...")
    X_train = extract_embeddings(train_df[content_col].tolist(), tokenizer, model, device, batch_size)
    print("Trích embedding cho tập val...")
    X_val = extract_embeddings(val_df[content_col].tolist(), tokenizer, model, device, batch_size)
    print("Trích embedding cho tập test...")
    X_test = extract_embeddings(test_df[content_col].tolist(), tokenizer, model, device, batch_size)

    y_train = train_df[label_col].values
    y_val = val_df[label_col].values
    y_test = test_df[label_col].values

    # Lưu embedding lại (tốn thời gian trích nhất, nên cache để khỏi
    # phải chạy lại BGE-M3 nếu chỉ muốn thử classifier khác)
    np.savez(
        f"{output_dir}/bge_m3_embeddings.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    print(f"Đã lưu embedding vào: {output_dir}/bge_m3_embeddings.npz")

    # --- Train classifier nhẹ lên trên embedding ---
    if classifier == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif classifier == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(256,), max_iter=500, random_state=RANDOM_STATE,
            early_stopping=True,
        )
    else:
        raise ValueError("classifier phải là 'logreg' hoặc 'mlp'")

    print(f"\nĐang train classifier ({classifier}) trên embedding BGE-M3...")
    clf.fit(X_train, y_train)

    # Đánh giá trên val (tham khảo) rồi test (số liệu chính thức)
    val_preds = clf.predict(X_val)
    print("\n=== Kết quả trên tập VAL ===")
    print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f} | "
          f"F1-macro: {f1_score(y_val, val_preds, average='macro'):.4f}")

    test_preds = clf.predict(X_test)
    print("\n=== Kết quả trên tập TEST (số liệu chính thức để so sánh) ===")
    print(classification_report(
        y_test, test_preds, target_names=[ID2LABEL[i] for i in range(3)], digits=4
    ))

    # --- Lưu classifier ---
    import joblib
    joblib.dump(clf, f"{output_dir}/classifier_{classifier}.joblib")
    print(f"Đã lưu classifier vào: {output_dir}/classifier_{classifier}.joblib")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BGE-M3 feature extraction + classifier cho phân loại cảm xúc 3 lớp"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="./bge_m3_sentiment_model")
    parser.add_argument("--content_col", default="content")
    parser.add_argument("--label_col", default="sentiment_label")
    parser.add_argument("--classifier", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_dir=args.output_dir,
        content_col=args.content_col,
        label_col=args.label_col,
        classifier=args.classifier,
        batch_size=args.batch_size,
    )