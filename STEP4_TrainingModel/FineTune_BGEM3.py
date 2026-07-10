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
pip install transformers torch scikit-learn pandas underthesea --break-system-packages
"""

import numpy as np
import pandas as pd
import torch
import sys
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from underthesea import word_tokenize

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

MODEL_NAME = "BAAI/bge-m3"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TOKENIZATION_INPUT = SCRIPT_DIR / "Tokenization" / "merch_datasets_student_sentiment_preprocessed.csv"
DEFAULT_CONTENT_COL = "content_preprocessed"
DEFAULT_LABEL_COL = "label"
WORD_SEGMENTED_COL = "word_segmented"
LOSS_LABELS_VI = {
    "train_loss": "Mất mát huấn luyện",
}
LEGACY_TOKENIZED_COLS = (
    "content_phobert_tokenized",
    "content_bgem3_tokenized",
    "content_simcse_tokenized",
)

ID2LABEL = {0: "tieu_cuc", 1: "trung_lap", 2: "tich_cuc"}

# GIỐNG HỆT train_phobert.py / train_simcse.py - bắt buộc để so sánh công bằng
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


def preprocess_for_bgem3(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text
    try:
        return word_tokenize(text, format="text")
    except Exception:
        return text


def read_dataframe(input_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path, encoding="utf-8-sig")
    return pd.read_excel(input_path)


def write_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(output_path, index=False)


def tokenize_dataset_file(
    input_path: str | Path = DEFAULT_TOKENIZATION_INPUT,
    output_path: str | Path | None = None,
    content_col: str = DEFAULT_CONTENT_COL,
    word_segmented_col: str = WORD_SEGMENTED_COL,
) -> Path:
    """Read dataset, tokenize content_col with underthesea, and save a new column."""
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path

    df = read_dataframe(input_path)
    if content_col not in df.columns:
        raise ValueError(f"Missing column '{content_col}'. Available columns: {list(df.columns)}")

    df = df.drop(columns=list(LEGACY_TOKENIZED_COLS), errors="ignore")
    df[word_segmented_col] = df[content_col].apply(preprocess_for_bgem3)
    write_dataframe(df, output_path)
    print(f"Saved word_segmented column '{word_segmented_col}' to: {output_path}")
    return output_path


def select_model_text_col(df: pd.DataFrame, content_col: str) -> str:
    if content_col == DEFAULT_CONTENT_COL and WORD_SEGMENTED_COL in df.columns:
        return WORD_SEGMENTED_COL
    return content_col


def load_and_split(input_path: str, content_col: str, label_col: str = DEFAULT_LABEL_COL):
    df = read_dataframe(input_path)

    missing_cols = [col for col in (content_col, label_col) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols}. Available columns: {list(df.columns)}")

    df = df.dropna(subset=[content_col, label_col]).copy()
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


def save_classifier_training_artifacts(clf, output_dir: str | Path) -> None:
    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    def jsonable(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value

    training_info = {
        "n_iter": jsonable(getattr(clf, "n_iter_", None)),
        "best_validation_score": jsonable(getattr(clf, "best_validation_score_", None)),
        "early_stopping": jsonable(getattr(clf, "early_stopping", None)),
        "n_iter_no_change": jsonable(getattr(clf, "n_iter_no_change", None)),
    }
    with (report_dir / "classifier_training_info.json").open("w", encoding="utf-8") as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)

    losses = getattr(clf, "loss_curve_", None)
    if not losses:
        return

    loss_history = pd.DataFrame(
        {
            "epoch": list(range(1, len(losses) + 1)),
            "train_loss": [float(loss) for loss in losses],
        }
    )
    loss_history.to_csv(report_dir / "loss_curve_points.csv", index=False, encoding="utf-8-sig")

    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        loss_history["epoch"],
        loss_history["train_loss"],
        marker="o",
        label=LOSS_LABELS_VI["train_loss"],
    )
    max_epoch = max(float(loss_history["epoch"].max()), 1.0)
    ax.set_xlim(left=0, right=max_epoch)
    if max_epoch <= 25:
        ax.set_xticks(range(0, int(np.ceil(max_epoch)) + 1))
    ax.set_title("Hàm mất mát bộ phân loại BGE-M3")
    ax.set_xlabel("Vòng huấn luyện")
    ax.set_ylabel("Giá trị mất mát")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "loss_curve.png", dpi=200)
    plt.close(fig)


def apply_feature_dropout(embeddings: np.ndarray, dropout: float, seed: int) -> np.ndarray:
    if dropout <= 0:
        return embeddings
    if dropout >= 1:
        raise ValueError("feature_dropout phai nho hon 1")
    rng = np.random.default_rng(seed)
    keep_mask = rng.random(embeddings.shape) >= dropout
    return (embeddings * keep_mask.astype(embeddings.dtype)) / (1.0 - dropout)


def run(input_path: str, output_dir: str, content_col: str = DEFAULT_CONTENT_COL,
        label_col: str = DEFAULT_LABEL_COL, classifier: str = "mlp",
        batch_size: int = 16, epochs: int = 50, min_epochs: int = 20, patience: int = 5,
        max_iter: int | None = None, logreg_c: float = 0.5,
        mlp_alpha: float = 1e-3, feature_dropout: float = 0.1,
        validation_fraction: float = 0.15):

    import os
    os.makedirs(output_dir, exist_ok=True)
    min_epochs = max(1, int(min_epochs))
    if epochs < min_epochs:
        print(f"[INFO] epochs={epochs} < min_epochs={min_epochs}; tu dong nang epochs len {min_epochs}.")
        epochs = min_epochs

    train_df, val_df, test_df = load_and_split(input_path, content_col, label_col)
    model_text_col = select_model_text_col(train_df, content_col)
    if model_text_col != content_col:
        print(f"Dùng cột tokenized đã lưu cho BGE-M3: {model_text_col}")

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
    X_train = extract_embeddings(train_df[model_text_col].tolist(), tokenizer, model, device, batch_size)
    print("Trích embedding cho tập val...")
    X_val = extract_embeddings(val_df[model_text_col].tolist(), tokenizer, model, device, batch_size)
    print("Trích embedding cho tập test...")
    X_test = extract_embeddings(test_df[model_text_col].tolist(), tokenizer, model, device, batch_size)

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
    classifier_max_iter = max(max_iter if max_iter is not None else epochs, min_epochs)

    if classifier == "logreg":
        clf = LogisticRegression(
            C=logreg_c,
            max_iter=classifier_max_iter,
            class_weight="balanced",
        )
    elif classifier == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(256,),
            max_iter=classifier_max_iter,
            alpha=mlp_alpha,
            validation_fraction=validation_fraction,
            random_state=RANDOM_STATE,
            early_stopping=patience > 0,
            # sklearn MLPClassifier khong co min_epochs rieng; dat n_iter_no_change >= min_epochs + patience
            # de chan viec early stop truoc moc toi thieu.
            n_iter_no_change=max(1, min_epochs + patience),
        )
    else:
        raise ValueError("classifier phải là 'logreg' hoặc 'mlp'")

    print(f"\nĐang train classifier ({classifier}) trên embedding BGE-M3...")
    X_train_classifier = apply_feature_dropout(X_train, feature_dropout, RANDOM_STATE)
    clf.fit(X_train_classifier, y_train)
    save_classifier_training_artifacts(clf, output_dir)

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
    parser.add_argument("--input", default=str(DEFAULT_TOKENIZATION_INPUT))
    parser.add_argument("--output_dir", default="./bge_m3_sentiment_model")
    parser.add_argument("--content_col", default=DEFAULT_CONTENT_COL)
    parser.add_argument("--label_col", default=DEFAULT_LABEL_COL)
    parser.add_argument("--classifier", choices=["logreg", "mlp"], default="mlp")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_epochs", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=None)
    parser.add_argument("--logreg_c", type=float, default=0.5)
    parser.add_argument("--mlp_alpha", type=float, default=1e-3)
    parser.add_argument("--feature_dropout", type=float, default=0.1)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--tokenize_only", action="store_true")
    parser.add_argument("--tokenized_output", default=None)
    parser.add_argument("--word_segmented_col", default=WORD_SEGMENTED_COL)
    args = parser.parse_args()

    if args.tokenize_only:
        tokenize_dataset_file(
            input_path=args.input,
            output_path=args.tokenized_output,
            content_col=args.content_col,
            word_segmented_col=args.word_segmented_col,
        )
    else:
        run(
            input_path=args.input,
            output_dir=args.output_dir,
            content_col=args.content_col,
            label_col=args.label_col,
            classifier=args.classifier,
            batch_size=args.batch_size,
            epochs=args.epochs,
            min_epochs=args.min_epochs,
            patience=args.patience,
            max_iter=args.max_iter,
            logreg_c=args.logreg_c,
            mlp_alpha=args.mlp_alpha,
            feature_dropout=args.feature_dropout,
            validation_fraction=args.validation_fraction,
        )
