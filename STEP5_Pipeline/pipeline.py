# -*- coding: utf-8 -*-
"""
End-to-end training pipeline for the STEP4 sentiment models.

Default behavior:
  - Read STEP4_TrainingModel/Tokenization/merch_datasets_student_sentiment_preprocessed.csv
  - Use word_segmented as text input and label as target
  - Create or reuse one stratified train/val/test split in STEP5_Pipeline/data
  - Train LSTM, PhoBERT, SimCSE, and BGE-M3
  - Export models, metrics, predictions, confusion matrices, training logs,
    and loss/metric plots.

Examples:
  python STEP5_Pipeline/pipeline.py
  python STEP5_Pipeline/pipeline.py --models lstm
  python STEP5_Pipeline/pipeline.py --models phobert simcse --transformer_epochs 3
  python STEP5_Pipeline/pipeline.py --models bgem3 --bgem3_classifier logreg
"""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import logging
import math
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from transformers import TrainerCallback as _TrainerCallbackBase
except Exception:
    _TrainerCallbackBase = object


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
STEP4_DIR = PROJECT_ROOT / "STEP4_TrainingModel"

DEFAULT_DATASET = STEP4_DIR / "Tokenization" / "merch_datasets_student_sentiment_preprocessed.csv"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"

ID2LABEL = {0: "tieu_cuc", 1: "trung_lap", 2: "tich_cuc"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
LABEL_IDS = list(ID2LABEL.keys())
ID2LABEL_VI = {0: "Tiêu cực", 1: "Trung lập", 2: "Tích cực"}

LOSS_LABELS_VI = {
    "train_loss": "Mất mát huấn luyện",
    "val_loss": "Mất mát xác thực",
    "eval_loss": "Mất mát đánh giá",
}
LOSS_COLUMNS = ("train_loss", "val_loss", "eval_loss")
METRIC_LABELS_VI = {
    "accuracy": "Tỷ lệ đúng",
    "precision_macro": "Độ chuẩn xác macro",
    "recall_macro": "Độ bao phủ macro",
    "f1_macro": "F1 macro",
    "f1_weighted": "F1 có trọng số",
}
MODEL_LABELS_VI = {
    "lstm": "LSTM",
    "phobert": "PhoBERT",
    "simcse": "SimCSE",
    "bgem3": "BGE-M3",
}
SPLIT_LABELS_VI = {
    "train": "Huấn luyện",
    "val": "Xác thực",
    "test": "Kiểm tra",
}
LABEL_NAME_VI = {name: ID2LABEL_VI[label_id] for label_id, name in ID2LABEL.items()}

LABEL_ALIASES = {
    "0": 0,
    "0.0": 0,
    "negative": 0,
    "neg": 0,
    "label_0": 0,
    "tieu_cuc": 0,
    "tieu cuc": 0,
    "1": 1,
    "1.0": 1,
    "neutral": 1,
    "neu": 1,
    "label_1": 1,
    "trung_lap": 1,
    "trung lap": 1,
    "2": 2,
    "2.0": 2,
    "positive": 2,
    "pos": 2,
    "label_2": 2,
    "tich_cuc": 2,
    "tich cuc": 2,
}

TRANSFORMER_SPECS = {
    "phobert": {
        "step4_file": STEP4_DIR / "FineTune_PhoBERT.py",
        "fallback_model_name": "vinai/phobert-base",
        "use_fast": False,
    },
    "simcse": {
        "step4_file": STEP4_DIR / "FineTune_SimCSE.py",
        "fallback_model_name": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
        "use_fast": False,
    },
}

BGEM3_SPEC = {
    "step4_file": STEP4_DIR / "FineTune_BGEM3.py",
    "fallback_model_name": "BAAI/bge-m3",
}

ALL_MODEL_KEYS = ("lstm", "phobert", "simcse", "bgem3")

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1
SPLIT_NAMES = ("train", "val", "test")


def build_logger(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_step4_constant(py_path: Path, constant_name: str, fallback: str) -> str:
    """Read a simple top-level constant from STEP4 scripts without importing them."""
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == constant_name:
                    value = ast.literal_eval(node.value)
                    if isinstance(value, str) and value.strip():
                        return value
    except Exception:
        return fallback
    return fallback


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    return torch.device(device_arg)


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_label(value: object) -> int:
    if pd.isna(value):
        raise ValueError("Missing label")

    if isinstance(value, (int, np.integer)):
        label_id = int(value)
        if label_id in ID2LABEL:
            return label_id

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        label_id = int(value)
        if label_id in ID2LABEL:
            return label_id

    key = re.sub(r"\s+", " ", str(value).strip().lower())
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]

    raise ValueError(
        f"Invalid label {value!r}. Expected 0/1/2 or aliases such as "
        "negative/neutral/positive."
    )


def read_table(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input file: {path}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_stratify_values(
    df: pd.DataFrame,
    split_strategy: str,
    logger: logging.Logger,
    stage_name: str,
) -> pd.Series:
    strategy_columns = {
        "label": ["label_id"],
        "label_data_type": ["label_id", "data_types"],
        "label_data_type_emotion": ["label_id", "data_types", "emotion"],
    }
    requested_columns = strategy_columns[split_strategy]
    available_columns = [col for col in requested_columns if col in df.columns]
    if available_columns != requested_columns:
        missing = [col for col in requested_columns if col not in df.columns]
        logger.warning(
            "Split strategy %s missing columns %s at %s; fallback to label only.",
            split_strategy,
            missing,
            stage_name,
        )
        available_columns = ["label_id"]

    values = df[available_columns].astype(str).agg("||".join, axis=1)
    counts = values.value_counts()
    if counts.empty or counts.min() < 2:
        if available_columns != ["label_id"]:
            logger.warning(
                "Split strategy %s creates rare strata at %s (min_count=%s); fallback to label only.",
                split_strategy,
                stage_name,
                int(counts.min()) if not counts.empty else 0,
            )
            return df["label_id"].astype(str)
        raise ValueError("Cannot stratify split because at least one label has fewer than 2 rows.")

    logger.info(
        "Using split strategy %s at %s with columns %s (%s strata, min_count=%s).",
        split_strategy,
        stage_name,
        available_columns,
        len(counts),
        int(counts.min()),
    )
    return values


def load_and_split_dataset(
    dataset_path: Path,
    text_col: str,
    label_col: str,
    val_size: float,
    test_size: float,
    split_strategy: str,
    seed: int,
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    df = read_table(dataset_path)
    missing_cols = [col for col in (text_col, label_col) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols}. Available columns: {list(df.columns)}")

    df = df.copy()
    df["_source_row_id"] = np.arange(len(df))
    df[text_col] = df[text_col].apply(clean_text)
    df = df[df[text_col].str.len() > 0].dropna(subset=[label_col]).copy()
    df["label_id"] = df[label_col].apply(normalize_label).astype(int)
    df["label_name"] = df["label_id"].map(ID2LABEL)

    label_counts = df["label_id"].value_counts().sort_index()
    logger.info("Loaded %s rows from %s", len(df), dataset_path)
    for label_id in LABEL_IDS:
        logger.info("Label %s (%s): %s rows", label_id, ID2LABEL[label_id], int(label_counts.get(label_id, 0)))

    missing_labels = [label_id for label_id in LABEL_IDS if label_id not in label_counts.index]
    if missing_labels:
        raise ValueError(f"Missing labels {missing_labels}; need all labels {LABEL_IDS}.")

    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("--val_size and --test_size must be > 0 and sum to < 1.")

    train_temp_stratify = build_stratify_values(df, split_strategy, logger, "train/temp")
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=train_temp_stratify,
    )
    relative_test_size = test_size / (val_size + test_size)
    val_test_stratify = build_stratify_values(temp_df, split_strategy, logger, "val/test")
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=seed,
        stratify=val_test_stratify,
    )

    splits = {
        "train": train_df.reset_index(drop=True).copy(),
        "val": val_df.reset_index(drop=True).copy(),
        "test": test_df.reset_index(drop=True).copy(),
    }
    for split_name, split_df in splits.items():
        split_df["split"] = split_name
        logger.info("%s rows: %s", split_name, len(split_df))

    return splits


def export_splits(splits: dict[str, pd.DataFrame], split_dir: Path, logger: logging.Logger) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in splits.items():
        write_csv(split_df, split_dir / f"{split_name}.csv")

    rows = []
    for split_name, split_df in splits.items():
        counts = split_df["label_id"].value_counts().sort_index()
        for label_id in LABEL_IDS:
            rows.append(
                {
                    "split": split_name,
                    "label_id": label_id,
                    "label_name": ID2LABEL[label_id],
                    "count": int(counts.get(label_id, 0)),
                    "ratio": float(counts.get(label_id, 0) / max(1, len(split_df))),
                }
            )

    distribution = pd.DataFrame(rows)
    write_csv(distribution, split_dir / "split_distribution.csv")
    plot_split_distribution(distribution, split_dir / "split_distribution.png")
    for extra_col in ("data_types", "emotion"):
        if all(extra_col in split_df.columns for split_df in splits.values()):
            extra_rows = []
            for split_name, split_df in splits.items():
                grouped = (
                    split_df.groupby([extra_col, "label_id"], dropna=False)
                    .size()
                    .reset_index(name="count")
                )
                for _, row in grouped.iterrows():
                    extra_rows.append(
                        {
                            "split": split_name,
                            extra_col: row[extra_col],
                            "label_id": int(row["label_id"]),
                            "label_name": ID2LABEL[int(row["label_id"])],
                            "count": int(row["count"]),
                        }
                    )
            write_csv(pd.DataFrame(extra_rows), split_dir / f"split_distribution_by_{extra_col}.csv")
    logger.info("Saved split files to %s", split_dir)


def split_file_map(data_dir: Path) -> dict[str, Path]:
    return {split_name: data_dir / f"{split_name}.csv" for split_name in SPLIT_NAMES}


def load_saved_splits(
    data_dir: Path,
    text_col: str,
    label_col: str,
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    splits = {}
    for split_name, split_path in split_file_map(data_dir).items():
        split_df = read_table(split_path).copy()
        missing_cols = [col for col in (text_col, label_col) if col not in split_df.columns]
        if missing_cols:
            raise ValueError(
                f"File {split_path} is missing columns {missing_cols}. "
                f"Available columns: {list(split_df.columns)}"
            )

        split_df[text_col] = split_df[text_col].apply(clean_text)
        split_df = split_df[split_df[text_col].str.len() > 0].dropna(subset=[label_col]).copy()
        if "label_id" not in split_df.columns:
            split_df["label_id"] = split_df[label_col].apply(normalize_label).astype(int)
        else:
            split_df["label_id"] = split_df["label_id"].apply(normalize_label).astype(int)
        split_df["label_name"] = split_df["label_id"].map(ID2LABEL)
        split_df["split"] = split_name
        splits[split_name] = split_df.reset_index(drop=True)

    for split_name, split_df in splits.items():
        logger.info("Loaded %s rows from cached %s split in %s", len(split_df), split_name, data_dir)
    return splits


def get_or_create_data_splits(
    dataset_path: Path,
    data_dir: Path,
    text_col: str,
    label_col: str,
    val_size: float,
    test_size: float,
    split_strategy: str,
    seed: int,
    rebuild_data: bool,
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    data_dir.mkdir(parents=True, exist_ok=True)
    split_paths = split_file_map(data_dir)
    existing = [name for name, path in split_paths.items() if path.exists()]
    missing = [name for name, path in split_paths.items() if not path.exists()]

    if len(existing) == len(SPLIT_NAMES) and not rebuild_data:
        config_path = data_dir / "data_config.json"
        cached_strategy = "label"
        if config_path.exists():
            try:
                cached_config = json.loads(config_path.read_text(encoding="utf-8"))
                cached_strategy = cached_config.get("split_strategy", "label")
            except Exception:
                cached_strategy = "unknown"
        if cached_strategy != split_strategy:
            raise ValueError(
                f"Cached split data in {data_dir} was built with split_strategy={cached_strategy!r}, "
                f"but current --split_strategy is {split_strategy!r}. "
                "Run with --rebuild_data to recreate splits, or pass a different --data_dir."
            )
        logger.info("Using cached train/val/test data from %s", data_dir)
        return load_saved_splits(data_dir, text_col, label_col, logger)

    if existing and missing and not rebuild_data:
        raise ValueError(
            f"Data folder {data_dir} has partial split files. "
            f"Existing: {existing}; missing: {missing}. "
            "Add the missing files or run with --rebuild_data to recreate all splits."
        )

    if rebuild_data and existing:
        logger.info("Rebuilding train/val/test split files in %s", data_dir)
    else:
        logger.info("Creating train/val/test split files in %s", data_dir)

    splits = load_and_split_dataset(
        dataset_path=dataset_path,
        text_col=text_col,
        label_col=label_col,
        val_size=val_size,
        test_size=test_size,
        split_strategy=split_strategy,
        seed=seed,
        logger=logger,
    )
    export_splits(splits, data_dir, logger)
    save_json(
        data_dir / "data_config.json",
        {
            "dataset": str(dataset_path),
            "text_col": text_col,
            "label_col": label_col,
            "val_size": val_size,
            "test_size": test_size,
            "split_strategy": split_strategy,
            "seed": seed,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "split_files": {name: str(path) for name, path in split_paths.items()},
        },
    )
    return splits


def compute_metric_dict(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def classification_report_dict(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, object]:
    return classification_report(
        y_true,
        y_pred,
        labels=LABEL_IDS,
        target_names=[ID2LABEL[i] for i in LABEL_IDS],
        output_dict=True,
        zero_division=0,
    )


def save_confusion_matrix_artifacts(y_true: Sequence[int], y_pred: Sequence[int], out_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{ID2LABEL[i]}" for i in LABEL_IDS],
        columns=[f"pred_{ID2LABEL[i]}" for i in LABEL_IDS],
    )
    write_csv(cm_df.reset_index().rename(columns={"index": "true_label"}), out_dir / "confusion_matrix.csv")

    if plt is None:
        return
    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm, out_dir / "confusion_matrix_normalized.png", normalize=True)


def plot_confusion_matrix(cm: np.ndarray, path: Path, normalize: bool) -> None:
    values = cm.astype(float)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, np.maximum(row_sums, 1.0))

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(values, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    tick_labels = [ID2LABEL_VI[i] for i in LABEL_IDS]
    ax.set(
        xticks=np.arange(len(tick_labels)),
        yticks=np.arange(len(tick_labels)),
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ylabel="Nhãn thật",
        xlabel="Nhãn dự đoán",
        title="Ma trận nhầm lẫn chuẩn hóa" if normalize else "Ma trận nhầm lẫn",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    threshold = values.max() / 2.0 if values.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = f"{values[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if values[i, j] > threshold else "black",
            )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_split_distribution(distribution: pd.DataFrame, path: Path) -> None:
    if plt is None:
        return
    plot_df = distribution.copy()
    plot_df["split"] = plot_df["split"].map(SPLIT_LABELS_VI).fillna(plot_df["split"])
    plot_df["label_name"] = plot_df["label_name"].map(LABEL_NAME_VI).fillna(plot_df["label_name"])
    pivot = plot_df.pivot(index="split", columns="label_name", values="count").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_title("Phân bố nhãn theo tập dữ liệu")
    ax.set_xlabel("Tập dữ liệu")
    ax.set_ylabel("Số dòng")
    ax.legend(title="Nhãn")
    fig = ax.get_figure()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def loss_curve_points(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["epoch", *LOSS_COLUMNS])

    normalizer = math.log(max(2, len(LABEL_IDS)))
    points = pd.DataFrame()
    points["epoch"] = history["epoch"]
    for column in LOSS_COLUMNS:
        if column not in history.columns:
            continue
        raw_values = pd.to_numeric(history[column], errors="coerce")
        points[f"raw_{column}"] = raw_values
        points[column] = raw_values / normalizer
    return points


def plot_epoch_history(history: pd.DataFrame, path: Path, title: str) -> None:
    if plt is None or history.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for column in LOSS_COLUMNS:
        if column not in history.columns:
            continue
        points = history[["epoch", column]].dropna().copy()
        if not points.empty:
            points["plot_epoch"] = pd.to_numeric(points["epoch"], errors="coerce") - 1
            points = points.dropna(subset=["plot_epoch"])
            points["plot_epoch"] = points["plot_epoch"].clip(lower=0)
        if not points.empty:
            ax.plot(points["plot_epoch"], points[column], marker="o", label=LOSS_LABELS_VI.get(column, column))
    ax.set_title(title)
    ax.set_xlabel("Vòng huấn luyện")
    ax.set_ylabel("Mất mát chuẩn hóa")
    max_epoch = pd.to_numeric(history["epoch"], errors="coerce").max()
    if pd.notna(max_epoch):
        max_plot_epoch = max(float(max_epoch) - 1.0, 0.0)
        axis_right = max(max_plot_epoch, 1.0)
        ax.set_xlim(left=0, right=axis_right)
        if max_plot_epoch <= 25:
            ax.set_xticks(range(0, int(math.ceil(axis_right)) + 1))
    ax.grid(True, alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def monitor_improved(current: float, best: float, monitor: str, min_delta: float) -> bool:
    if monitor.endswith("loss"):
        return current < best - min_delta
    return current > best + min_delta


def initial_monitor_score(monitor: str) -> float:
    return math.inf if monitor.endswith("loss") else -math.inf

class MinimumEpochEarlyStoppingCallback(_TrainerCallbackBase):
    """Early stopping co min_epochs: van theo doi best metric tu dau, nhung chi tinh patience sau min_epochs."""

    def __init__(
        self,
        early_stopping_patience: int,
        early_stopping_threshold: float = 0.0,
        min_epochs: int = 20,
        metric_name: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        self.early_stopping_patience = max(1, int(early_stopping_patience))
        self.early_stopping_threshold = float(early_stopping_threshold)
        self.min_epochs = max(1, int(min_epochs))
        self.metric_name = metric_name
        self.greater_is_better = bool(greater_is_better)
        self.best_score: float | None = None
        self.bad_epochs = 0

    def __getattr__(self, name: str):
        if name.startswith("on_"):
            return self._noop_event
        raise AttributeError(name)

    def _noop_event(self, args, state, control, **kwargs):
        return control

    def _is_improvement(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self.greater_is_better:
            return current > self.best_score + self.early_stopping_threshold
        return current < self.best_score - self.early_stopping_threshold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        metric_value = metrics.get(self.metric_name)
        if metric_value is None and not self.metric_name.startswith("eval_"):
            metric_value = metrics.get(f"eval_{self.metric_name}")
        if metric_value is None:
            return control

        current = float(metric_value)
        if self._is_improvement(current):
            self.best_score = current
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        # Van dem bad_epochs tu dau, nhung khong cho dung truoc min_epochs.
        current_epoch = float(state.epoch or 0.0)
        if current_epoch < self.min_epochs:
            return control

        if self.bad_epochs >= self.early_stopping_patience:
            control.should_training_stop = True
        return control



def safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def epoch_bucket(value: object) -> int | None:
    epoch = safe_float(value)
    if epoch is None or epoch <= 0:
        return None
    return max(1, int(math.ceil(epoch - 1e-8)))


def transformer_epoch_loss_history(history: list[dict[str, object]]) -> pd.DataFrame:
    train_losses: dict[int, list[float]] = {}
    eval_losses: dict[int, list[float]] = {}
    fallback_train_loss: dict[int, float] = {}
    seen_epochs: list[int] = []

    for row in history:
        epoch = epoch_bucket(row.get("epoch"))
        if epoch is None:
            continue
        seen_epochs.append(epoch)

        loss = safe_float(row.get("loss"))
        if loss is not None:
            train_losses.setdefault(epoch, []).append(loss)

        train_loss = safe_float(row.get("train_loss"))
        if train_loss is not None:
            fallback_train_loss[epoch] = train_loss

        eval_loss = safe_float(row.get("eval_loss"))
        if eval_loss is not None:
            eval_losses.setdefault(epoch, []).append(eval_loss)

    if not train_losses and fallback_train_loss:
        train_losses = {epoch: [loss] for epoch, loss in fallback_train_loss.items()}

    if not train_losses and not eval_losses:
        return pd.DataFrame()

    max_epoch = max(seen_epochs + list(train_losses.keys()) + list(eval_losses.keys()))
    rows = []
    for epoch in range(1, max_epoch + 1):
        row: dict[str, float | int] = {"epoch": epoch}
        if epoch in train_losses:
            row["train_loss"] = float(np.mean(train_losses[epoch]))
        if epoch in eval_losses:
            row["val_loss"] = float(np.mean(eval_losses[epoch]))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_transformer_log_history(history: list[dict[str, object]], path: Path, title: str) -> pd.DataFrame:
    loss_history = transformer_epoch_loss_history(history)
    plot_epoch_history(loss_history, path, title)
    return loss_history


def metric_scalar_payload(metrics: dict[str, object], split: str = "val") -> dict[str, object]:
    payload: dict[str, object] = {"split": split}
    for name, value in metrics.items():
        if name in {"labels", "preds", "probs"}:
            continue
        if name.startswith("eval_"):
            name = name.removeprefix("eval_")
        scalar = safe_float(value)
        if scalar is not None:
            payload[name] = scalar
    return payload


def save_training_history_artifacts(model_dir: Path, history_df: pd.DataFrame, title: str) -> None:
    reports_dir = model_dir / "reports"
    if history_df.empty:
        history_df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])
    history_df = history_df.copy()
    write_csv(history_df, reports_dir / "train_history.csv")
    history_json = history_df.astype(object).where(pd.notna(history_df), None).to_dict(orient="records")
    save_json(reports_dir / "train_history.json", history_json)
    curve_df = loss_curve_points(history_df)
    write_csv(curve_df, reports_dir / "loss_curve_points.csv")
    plot_epoch_history(curve_df, reports_dir / "loss_curve.png", title)


def save_validation_metrics_artifacts(model_dir: Path, metrics: dict[str, object], split: str = "val") -> dict[str, object]:
    payload = metric_scalar_payload(metrics, split=split)
    reports_dir = model_dir / "reports"
    save_json(reports_dir / "validation_metrics.json", payload)
    write_csv(pd.DataFrame([payload]), reports_dir / "validation_metrics.csv")
    return payload


def relative_artifact_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def label_mapping_payload() -> dict[str, object]:
    return {
        "id2label": {str(label_id): label_name for label_id, label_name in ID2LABEL.items()},
        "label2id": LABEL2ID,
    }


def write_model_readme(model_save_dir: Path, model_key: str, loader_type: str) -> None:
    text = "\n".join(
        [
            f"# STEP5 {MODEL_LABELS_VI.get(model_key, model_key)} model",
            "",
            "This folder is the deployment/load directory for the trained model.",
            "",
            "Important files:",
            "- config.json: model architecture and label metadata",
            "- deployment_config.json: STEP5 loader contract",
            "- label_mapping.json: id2label and label2id",
            "- preprocessor_config.json: text preprocessing contract",
            "",
            f"Loader type: {loader_type}",
            "",
            "Transformer models can be loaded with Hugging Face AutoModel/AutoTokenizer.",
            "LSTM and BGE-M3 use the STEP5 deployment_config.json to locate their weights/classifier.",
            "",
        ]
    )
    model_save_dir.mkdir(parents=True, exist_ok=True)
    (model_save_dir / "README.md").write_text(text, encoding="utf-8")


def save_deployment_package(
    model_dir: Path,
    model_key: str,
    loader_type: str,
    framework: str,
    architecture: str,
    text_col: str,
    max_length: int,
    artifact_files: dict[str, Path | str | None],
    base_model_name: str | None = None,
    model_config: dict[str, object] | None = None,
    metrics: dict[str, object] | None = None,
    write_hf_config: bool = True,
) -> dict[str, object]:
    model_save_dir = model_dir / "model"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        name: relative_artifact_path(Path(path), model_save_dir) if path is not None else None
        for name, path in artifact_files.items()
    }
    labels = label_mapping_payload()
    config = {
        "model_type": f"step5_{model_key}",
        "architectures": [architecture],
        "pipeline_tag": "text-classification",
        "task": "sentiment-classification",
        "framework": framework,
        "loader_type": loader_type,
        "base_model_name_or_path": base_model_name,
        "text_column": text_col,
        "max_length": max_length,
        **labels,
    }
    if model_config:
        config["step5_model_config"] = model_config
    if write_hf_config:
        save_json(model_save_dir / "config.json", config)

    preprocessor_config = {
        "text_column": text_col,
        "cleaning": "clean_text: strip and collapse whitespace",
        "tokenizer": {
            "type": loader_type,
            "max_length": max_length,
            "base_model_name_or_path": base_model_name,
        },
    }
    deployment_config = {
        "format": "step5-hf-like",
        "format_version": 1,
        "model_key": model_key,
        "loader_type": loader_type,
        "framework": framework,
        "model_dir": ".",
        "artifacts": artifacts,
        "config_file": "config.json",
        "label_mapping_file": "label_mapping.json",
        "preprocessor_config_file": "preprocessor_config.json",
        "reports_dir": relative_artifact_path(model_dir / "reports", model_save_dir),
        "logs_dir": relative_artifact_path(model_dir / "logs", model_save_dir),
    }
    if metrics:
        deployment_config["metrics"] = {
            name: value
            for name, value in metrics.items()
            if name
            in {
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_weighted",
                "best_epoch",
                "best_val_f1_macro",
                "best_val_loss",
                "test_loss",
            }
        }

    save_json(model_save_dir / "label_mapping.json", labels)
    save_json(model_save_dir / "preprocessor_config.json", preprocessor_config)
    save_json(model_save_dir / "deployment_config.json", deployment_config)
    save_json(
        model_save_dir / "model_index.json",
        {
            "model": model_key,
            "pipeline_tag": "text-classification",
            "library_name": framework,
            "tags": ["step5", "sentiment-classification", model_key],
            "metrics": deployment_config.get("metrics", {}),
        },
    )
    write_model_readme(model_save_dir, model_key, loader_type)
    return deployment_config


def plot_model_comparison(metrics_df: pd.DataFrame, path: Path) -> None:
    if plt is None or metrics_df.empty:
        return
    metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    available = [col for col in metric_cols if col in metrics_df.columns]
    if not available:
        return
    plot_df = metrics_df.set_index("model")[available].rename(columns=METRIC_LABELS_VI)
    ax = plot_df.plot(kind="bar", figsize=(10, 6), rot=0)
    ax.set_ylim(0, 1)
    ax.set_title("So sánh mô hình trên tập kiểm tra")
    ax.set_xlabel("Mô hình")
    ax.set_ylabel("Điểm số")
    ax.legend(loc="lower right", title="Chỉ số")
    fig = ax.get_figure()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def save_evaluation_artifacts(
    model_dir: Path,
    model_name: str,
    test_df: pd.DataFrame,
    text_col: str,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probs: np.ndarray | None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    metrics = compute_metric_dict(y_true, y_pred)
    report = classification_report_dict(y_true, y_pred)

    payload = {
        "model": model_name,
        **metrics,
        "classification_report": report,
    }
    if extra:
        payload.update(extra)

    reports_dir = model_dir / "reports"
    save_json(reports_dir / "metrics.json", payload)
    write_csv(pd.DataFrame([{k: v for k, v in payload.items() if k != "classification_report"}]), reports_dir / "metrics.csv")
    write_csv(pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"}), reports_dir / "classification_report.csv")
    save_confusion_matrix_artifacts(y_true, y_pred, reports_dir)

    pred_df = test_df.copy()
    pred_df["true_label_id"] = list(map(int, y_true))
    pred_df["true_label_name"] = [ID2LABEL[int(label)] for label in y_true]
    pred_df["pred_label_id"] = list(map(int, y_pred))
    pred_df["pred_label_name"] = [ID2LABEL[int(label)] for label in y_pred]
    if probs is not None:
        for label_id in LABEL_IDS:
            pred_df[f"prob_{ID2LABEL[label_id]}"] = probs[:, label_id]
    cols = [text_col, "true_label_id", "true_label_name", "pred_label_id", "pred_label_name"]
    remaining = [col for col in pred_df.columns if col not in cols]
    write_csv(pred_df[cols + remaining], reports_dir / "test_predictions.csv")

    return payload


def tokenize_lstm_text(text: str) -> list[str]:
    tokens = clean_text(text).split()
    return tokens if tokens else [UNK_TOKEN]


def build_vocab(texts: Iterable[str], max_vocab: int, min_freq: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize_lstm_text(text))

    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, freq in counter.most_common(max(0, max_vocab - len(vocab))):
        if freq < min_freq:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_lstm_text(text: str, vocab: dict[str, int], max_length: int) -> tuple[list[int], int]:
    tokens = tokenize_lstm_text(text)[:max_length]
    ids = [vocab.get(token, UNK_ID) for token in tokens]
    length = max(1, len(ids))
    if not ids:
        ids = [UNK_ID]
    if len(ids) < max_length:
        ids.extend([PAD_ID] * (max_length - len(ids)))
    return ids, min(length, max_length)


class LSTMSentimentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, vocab: dict[str, int], max_length: int):
        self.samples = []
        for text, label in zip(df[text_col].tolist(), df["label_id"].tolist()):
            ids, length = encode_lstm_text(text, vocab, max_length)
            self.samples.append((ids, length, int(label), clean_text(text)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        ids, length, label, text = self.samples[index]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "lengths": torch.tensor(length, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
            "text": text,
        }


class BiLSTMSentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(last_hidden))


def make_lstm_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def class_weights(labels: Iterable[int], device: torch.device) -> torch.Tensor:
    counts = Counter(int(label) for label in labels)
    total = sum(counts.values())
    weights = [total / (len(LABEL_IDS) * max(1, counts.get(label_id, 0))) for label_id in LABEL_IDS]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def cross_entropy_loss_parts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> tuple[float, float]:
    losses = nn.functional.cross_entropy(
        logits,
        labels,
        weight=weight,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    if weight is None:
        denominator = float(labels.numel())
    else:
        denominator = float(weight.detach()[labels].sum().detach().cpu().item())
    return float(losses.sum().detach().cpu().item()), denominator


def train_lstm_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    loss_denom = 0.0
    unweighted_loss_sum = 0.0
    unweighted_loss_count = 0.0
    y_true = []
    y_pred = []
    loss_weight = getattr(criterion, "weight", None)
    label_smoothing = float(getattr(criterion, "label_smoothing", 0.0) or 0.0)

    for batch in tqdm(loader, desc="LSTM train", leave=False):
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        batch_loss_sum, batch_loss_denom = cross_entropy_loss_parts(
            logits, labels, loss_weight, label_smoothing
        )
        batch_unweighted_sum, batch_unweighted_count = cross_entropy_loss_parts(
            logits, labels, None, label_smoothing
        )
        loss_sum += batch_loss_sum
        loss_denom += batch_loss_denom
        unweighted_loss_sum += batch_unweighted_sum
        unweighted_loss_count += batch_unweighted_count
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    metrics = compute_metric_dict(y_true, y_pred)
    metrics["loss"] = loss_sum / loss_denom if loss_denom else 0.0
    metrics["unweighted_loss"] = unweighted_loss_sum / unweighted_loss_count if unweighted_loss_count else 0.0
    return metrics


@torch.no_grad()
def evaluate_lstm(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    loss_sum = 0.0
    loss_denom = 0.0
    unweighted_loss_sum = 0.0
    unweighted_loss_count = 0.0
    y_true = []
    y_pred = []
    probs = []
    texts = []
    loss_weight = getattr(criterion, "weight", None)
    label_smoothing = float(getattr(criterion, "label_smoothing", 0.0) or 0.0)

    for batch in tqdm(loader, desc="LSTM eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        batch_probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(batch_probs, dim=1)

        batch_loss_sum, batch_loss_denom = cross_entropy_loss_parts(
            logits, labels, loss_weight, label_smoothing
        )
        batch_unweighted_sum, batch_unweighted_count = cross_entropy_loss_parts(
            logits, labels, None, label_smoothing
        )
        loss_sum += batch_loss_sum
        loss_denom += batch_loss_denom
        unweighted_loss_sum += batch_unweighted_sum
        unweighted_loss_count += batch_unweighted_count
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        probs.extend(batch_probs.detach().cpu().tolist())
        texts.extend(batch["text"])

    metrics = compute_metric_dict(y_true, y_pred)
    metrics["loss"] = loss_sum / loss_denom if loss_denom else 0.0
    metrics["unweighted_loss"] = unweighted_loss_sum / unweighted_loss_count if unweighted_loss_count else 0.0
    return {
        **metrics,
        "labels": y_true,
        "preds": y_pred,
        "probs": np.asarray(probs, dtype=np.float32),
        "texts": texts,
    }


def torch_load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_lstm_model(
    args: argparse.Namespace,
    splits: dict[str, pd.DataFrame],
    run_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    model_key = "lstm"
    model_dir = run_dir / "models" / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(f"pipeline.{model_key}", model_dir / "training.log")
    logger.info("Starting LSTM training on %s", device)

    vocab = build_vocab(splits["train"][args.text_col].tolist(), args.lstm_max_vocab, args.lstm_min_freq)
    model_config = {
        "vocab_size": len(vocab),
        "embedding_dim": args.lstm_embedding_dim,
        "hidden_size": args.lstm_hidden_size,
        "num_layers": args.lstm_num_layers,
        "num_classes": len(LABEL_IDS),
        "dropout": args.lstm_dropout,
        "max_length": args.lstm_max_length,
    }
    save_json(model_dir / "model" / "vocab.json", vocab)
    save_json(model_dir / "model" / "label_mapping.json", {"id2label": ID2LABEL, "label2id": LABEL2ID})
    save_json(
        model_dir / "training_config.json",
        {
            **vars(args),
            **model_config,
            "loss_function": {
                "name": "CrossEntropyLoss",
                "class_weight": None if args.no_class_weight else "balanced",
                "label_smoothing": args.lstm_label_smoothing,
            },
            "regularization": {
                "dropout": args.lstm_dropout,
                "weight_decay": args.lstm_weight_decay,
                "grad_clip": args.lstm_grad_clip,
            },
        },
    )

    train_ds = LSTMSentimentDataset(splits["train"], args.text_col, vocab, args.lstm_max_length)
    val_ds = LSTMSentimentDataset(splits["val"], args.text_col, vocab, args.lstm_max_length)
    test_ds = LSTMSentimentDataset(splits["test"], args.text_col, vocab, args.lstm_max_length)

    train_loader = make_lstm_loader(train_ds, args.lstm_batch_size, True, args.num_workers)
    val_loader = make_lstm_loader(val_ds, args.lstm_batch_size, False, args.num_workers)
    test_loader = make_lstm_loader(test_ds, args.lstm_batch_size, False, args.num_workers)

    model = BiLSTMSentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=args.lstm_embedding_dim,
        hidden_size=args.lstm_hidden_size,
        num_layers=args.lstm_num_layers,
        num_classes=len(LABEL_IDS),
        dropout=args.lstm_dropout,
    ).to(device)

    weights = None if args.no_class_weight else class_weights(splits["train"]["label_id"].tolist(), device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.lstm_label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lstm_lr,
        weight_decay=args.lstm_weight_decay,
    )

    history_rows = []
    monitor = args.lstm_monitor
    best_monitor_score = initial_monitor_score(monitor)
    best_monitor_name = monitor
    max_val_f1 = -1.0
    best_checkpoint_val_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    checkpoint_path = model_dir / "model" / "lstm_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.lstm_epochs + 1):
        train_metrics = train_lstm_one_epoch(
            model, train_loader, optimizer, criterion, device, args.lstm_grad_clip
        )
        val_metrics = evaluate_lstm(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_unweighted_loss": train_metrics["unweighted_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision_macro": train_metrics["precision_macro"],
            "train_recall_macro": train_metrics["recall_macro"],
            "train_f1_macro": train_metrics["f1_macro"],
            "train_f1_weighted": train_metrics["f1_weighted"],
            "val_loss": val_metrics["loss"],
            "val_unweighted_loss": val_metrics["unweighted_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_weighted": val_metrics["f1_weighted"],
        }
        history_rows.append(row)
        logger.info(
            "Epoch %s/%s | train_loss=%.4f train_f1_macro=%.4f | val_loss=%.4f val_f1_macro=%.4f",
            epoch,
            args.lstm_epochs,
            row["train_loss"],
            row["train_f1_macro"],
            row["val_loss"],
            row["val_f1_macro"],
        )
        max_val_f1 = max(max_val_f1, row["val_f1_macro"])

        current_monitor_score = float(row[monitor])
        if monitor_improved(current_monitor_score, best_monitor_score, monitor, args.lstm_min_delta):
            best_monitor_score = current_monitor_score
            best_epoch = epoch
            best_checkpoint_val_f1 = row["val_f1_macro"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "model_config": model_config,
                    "id2label": ID2LABEL,
                    "label2id": LABEL2ID,
                    "best_epoch": best_epoch,
                    "best_monitor": best_monitor_name,
                    "best_monitor_score": best_monitor_score,
                    "best_val_f1_macro": best_checkpoint_val_f1,
                    "max_val_f1_macro": max_val_f1,
                    "best_val_loss": row["val_loss"],
                    "best_val_unweighted_loss": row["val_unweighted_loss"],
                },
                checkpoint_path,
            )
            logger.info(
                "Saved best LSTM checkpoint to %s (%s=%.4f)",
                checkpoint_path,
                best_monitor_name,
                best_monitor_score,
            )
        else:
            # Khong tinh patience truoc/sat min_epochs; chi bat dau dem tu epoch > min_epochs.
            if epoch > args.lstm_min_epochs:
                bad_epochs += 1
                if args.lstm_patience > 0 and bad_epochs >= args.lstm_patience:
                    logger.info(
                        "Early stopping after %s bad epochs without %s improvement, after minimum %s epochs.",
                        bad_epochs,
                        best_monitor_name,
                        args.lstm_min_epochs,
                    )
                    break
            else:
                bad_epochs = 0

    history_df = pd.DataFrame(history_rows)
    save_training_history_artifacts(model_dir, history_df, "Hàm mất mát LSTM")

    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics = evaluate_lstm(model, val_loader, criterion, device)
    validation_payload = save_validation_metrics_artifacts(model_dir, validation_metrics)
    test_metrics = evaluate_lstm(model, test_loader, criterion, device)
    payload = save_evaluation_artifacts(
        model_dir=model_dir,
        model_name=model_key,
        test_df=splits["test"],
        text_col=args.text_col,
        y_true=test_metrics["labels"],
        y_pred=test_metrics["preds"],
        probs=test_metrics["probs"],
        extra={
            "best_epoch": best_epoch,
            "best_monitor": best_monitor_name,
            "best_monitor_score": best_monitor_score,
            "best_val_f1_macro": best_checkpoint_val_f1,
            "max_val_f1_macro": max_val_f1,
            "best_val_loss": validation_payload.get("loss"),
            "best_val_unweighted_loss": validation_payload.get("unweighted_loss"),
            "test_loss": test_metrics["loss"],
            "test_unweighted_loss": test_metrics["unweighted_loss"],
            "model_path": str(model_dir / "model"),
            "weights_path": str(checkpoint_path),
        },
    )
    save_deployment_package(
        model_dir=model_dir,
        model_key=model_key,
        loader_type="step5_lstm",
        framework="pytorch",
        architecture="BiLSTMSentimentClassifier",
        text_col=args.text_col,
        max_length=args.lstm_max_length,
        artifact_files={
            "weights": checkpoint_path,
            "vocab": model_dir / "model" / "vocab.json",
            "training_config": model_dir / "training_config.json",
            "metrics": model_dir / "reports" / "metrics.json",
        },
        model_config=model_config,
        metrics=payload,
        write_hf_config=True,
    )
    logger.info("Finished LSTM. Test f1_macro=%.4f", payload["f1_macro"])
    return payload


@torch.no_grad()
def extract_bgem3_embeddings(
    texts: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    desc: str,
) -> np.ndarray:
    embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        batch_texts = [clean_text(text) for text in texts[start : start + batch_size]]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0]
        cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
        embeddings.append(cls_embeddings.detach().cpu().numpy().astype(np.float32))

    if not embeddings:
        raise ValueError(f"No rows available while extracting embeddings for {desc}.")
    return np.concatenate(embeddings, axis=0)


def build_bgem3_classifier(args: argparse.Namespace):
    max_iter = args.bgem3_max_iter if args.bgem3_max_iter is not None else args.bgem3_epochs

    if args.bgem3_classifier == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=args.bgem3_logreg_c,
            max_iter=max_iter,
            class_weight=None if args.no_class_weight else "balanced",
            random_state=args.seed,
        )

    if args.bgem3_classifier == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(
            hidden_layer_sizes=(args.bgem3_mlp_hidden_size,),
            max_iter=1,
            alpha=args.bgem3_mlp_alpha,
            random_state=args.seed,
            early_stopping=False,
        )

    raise ValueError("--bgem3_classifier must be 'logreg' or 'mlp'.")


def classifier_probabilities(classifier, embeddings: np.ndarray) -> np.ndarray | None:
    if not hasattr(classifier, "predict_proba"):
        return None

    raw_probs = classifier.predict_proba(embeddings)
    probs = np.zeros((len(embeddings), len(LABEL_IDS)), dtype=np.float32)
    for source_idx, class_id in enumerate(classifier.classes_):
        label_id = int(class_id)
        if label_id in LABEL_IDS:
            probs[:, label_id] = raw_probs[:, source_idx]
    return probs


def classifier_log_loss(classifier, embeddings: np.ndarray, labels: np.ndarray) -> float | None:
    probs = classifier_probabilities(classifier, embeddings)
    if probs is None:
        return None
    return float(log_loss(labels, probs, labels=LABEL_IDS))


def evaluate_bgem3_classifier(classifier, embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    preds = classifier.predict(embeddings).astype(int).tolist()
    metrics = compute_metric_dict(labels.tolist(), preds)
    loss_value = classifier_log_loss(classifier, embeddings, labels)
    if loss_value is not None:
        metrics["loss"] = loss_value
    return metrics


def prefixed_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    rows = {}
    for name, value in metrics.items():
        if name == "loss":
            rows[f"{prefix}_loss"] = value
        else:
            rows[f"{prefix}_{name}"] = value
    return rows


def train_bgem3_classifier(
    classifier,
    args: argparse.Namespace,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    logger: logging.Logger,
) -> tuple[object, pd.DataFrame, int, dict[str, float]]:
    if args.bgem3_classifier != "mlp":
        classifier.fit(train_embeddings, train_labels)
        train_metrics = evaluate_bgem3_classifier(classifier, train_embeddings, train_labels)
        val_metrics = evaluate_bgem3_classifier(classifier, val_embeddings, val_labels)
        history_row = {
            "epoch": 1,
            **prefixed_metrics(train_metrics, "train"),
            **prefixed_metrics(val_metrics, "val"),
        }
        return classifier, pd.DataFrame([history_row]), 1, val_metrics

    history_rows = []
    best_epoch = 0
    best_val_metrics: dict[str, float] = {}
    best_classifier = None
    best_score = -math.inf
    epochs_without_improvement = 0
    classes = np.asarray(LABEL_IDS, dtype=int)

    for epoch in range(1, args.bgem3_epochs + 1):
        classifier.partial_fit(train_embeddings, train_labels, classes=classes)

        train_metrics = evaluate_bgem3_classifier(classifier, train_embeddings, train_labels)
        val_metrics = evaluate_bgem3_classifier(classifier, val_embeddings, val_labels)
        history_rows.append(
            {
                "epoch": epoch,
                **prefixed_metrics(train_metrics, "train"),
                **prefixed_metrics(val_metrics, "val"),
            }
        )
        logger.info(
            "Epoch %s/%s | train_loss=%.4f train_f1_macro=%.4f | val_loss=%.4f val_f1_macro=%.4f",
            epoch,
            args.bgem3_epochs,
            train_metrics.get("loss", float("nan")),
            train_metrics["f1_macro"],
            val_metrics.get("loss", float("nan")),
            val_metrics["f1_macro"],
        )

        score = val_metrics["f1_macro"]
        improved = score > best_score + args.bgem3_min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            best_val_metrics = val_metrics.copy()
            best_classifier = copy.deepcopy(classifier)
            epochs_without_improvement = 0
        elif epoch >= args.bgem3_min_epochs:
            epochs_without_improvement += 1

        if (
            args.bgem3_patience > 0
            and epoch >= args.bgem3_min_epochs
            and epochs_without_improvement >= args.bgem3_patience
        ):
            logger.info(
                "Early stopping BGE-M3 MLP at epoch %s; best epoch=%s best_val_f1_macro=%.4f",
                epoch,
                best_epoch,
                best_score,
            )
            break

    if best_classifier is not None:
        classifier = best_classifier
    if not best_val_metrics:
        best_val_metrics = evaluate_bgem3_classifier(classifier, val_embeddings, val_labels)
    return classifier, pd.DataFrame(history_rows), best_epoch, best_val_metrics


def apply_feature_dropout(embeddings: np.ndarray, dropout: float, seed: int) -> np.ndarray:
    if dropout <= 0:
        return embeddings
    if dropout >= 1:
        raise ValueError("Feature dropout must be < 1.")
    rng = np.random.default_rng(seed)
    keep_mask = rng.random(embeddings.shape) >= dropout
    return (embeddings * keep_mask.astype(embeddings.dtype)) / (1.0 - dropout)


def run_bgem3_model(
    args: argparse.Namespace,
    splits: dict[str, pd.DataFrame],
    run_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    import joblib
    from transformers import AutoModel, AutoTokenizer

    model_key = "bgem3"
    model_name = read_step4_constant(
        BGEM3_SPEC["step4_file"],
        "MODEL_NAME",
        BGEM3_SPEC["fallback_model_name"],
    )
    model_dir = run_dir / "models" / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(f"pipeline.{model_key}", model_dir / "training.log")
    logger.info("Starting BGE-M3 feature extraction from checkpoint: %s", model_name)

    save_json(
        model_dir / "training_config.json",
        {
            **vars(args),
            "model_key": model_key,
            "model_name": model_name,
            "step4_file": str(BGEM3_SPEC["step4_file"]),
            "classifier": args.bgem3_classifier,
            "loss_function": {
                "logreg": "LogisticRegression multinomial cross-entropy",
                "mlp": "MLPClassifier cross-entropy",
            }[args.bgem3_classifier],
            "regularization": {
                "class_weight": None if args.no_class_weight else "balanced",
                "logreg_c": args.bgem3_logreg_c,
                "mlp_alpha_l2": args.bgem3_mlp_alpha,
                "feature_dropout": args.bgem3_feature_dropout,
            },
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()

    split_texts = {
        split_name: splits[split_name][args.text_col].apply(clean_text).tolist()
        for split_name in SPLIT_NAMES
    }
    split_labels = {
        split_name: splits[split_name]["label_id"].astype(int).to_numpy()
        for split_name in SPLIT_NAMES
    }

    split_embeddings = {}
    for split_name in SPLIT_NAMES:
        logger.info("Extracting BGE-M3 embeddings for %s split (%s rows)", split_name, len(split_texts[split_name]))
        split_embeddings[split_name] = extract_bgem3_embeddings(
            texts=split_texts[split_name],
            tokenizer=tokenizer,
            model=encoder,
            device=device,
            batch_size=args.bgem3_batch_size,
            max_length=args.bgem3_max_length,
            desc=f"BGE-M3 {split_name}",
        )

    model_save_dir = model_dir / "model"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = model_save_dir / "bge_m3_embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        X_train=split_embeddings["train"],
        y_train=split_labels["train"],
        X_val=split_embeddings["val"],
        y_val=split_labels["val"],
        X_test=split_embeddings["test"],
        y_test=split_labels["test"],
    )

    classifier_train_embeddings = apply_feature_dropout(
        split_embeddings["train"],
        args.bgem3_feature_dropout,
        args.seed,
    )
    classifier = build_bgem3_classifier(args)
    logger.info("Training BGE-M3 %s classifier", args.bgem3_classifier)
    classifier, classifier_history, best_epoch, val_metrics = train_bgem3_classifier(
        classifier=classifier,
        args=args,
        train_embeddings=classifier_train_embeddings,
        train_labels=split_labels["train"],
        val_embeddings=split_embeddings["val"],
        val_labels=split_labels["val"],
        logger=logger,
    )

    save_training_history_artifacts(
        model_dir,
        classifier_history,
        "Hàm mất mát bộ phân loại BGE-M3",
    )

    classifier_path = model_save_dir / f"classifier_{args.bgem3_classifier}.joblib"
    joblib.dump(classifier, classifier_path)
    save_json(model_save_dir / "label_mapping.json", {"id2label": ID2LABEL, "label2id": LABEL2ID})
    save_json(
        model_save_dir / "model_info.json",
        {
            "model_checkpoint": model_name,
            "classifier_path": str(classifier_path),
            "embeddings_path": str(embeddings_path),
            "embedding_strategy": "last_hidden_state_cls_l2_normalized",
            "classifier": args.bgem3_classifier,
            "classifier_regularization": {
                "logreg_c": args.bgem3_logreg_c,
                "mlp_alpha": args.bgem3_mlp_alpha,
                "feature_dropout": args.bgem3_feature_dropout,
                "validation_fraction": args.bgem3_validation_fraction,
                "early_stopping_patience": args.bgem3_patience,
                "min_epochs": args.bgem3_min_epochs,
                "min_delta": args.bgem3_min_delta,
            },
            "best_epoch": best_epoch,
        },
    )

    validation_payload = save_validation_metrics_artifacts(model_dir, val_metrics)

    y_pred = classifier.predict(split_embeddings["test"]).astype(int).tolist()
    probs = classifier_probabilities(classifier, split_embeddings["test"])
    test_loss = classifier_log_loss(classifier, split_embeddings["test"], split_labels["test"])
    extra_payload = {
        "model_checkpoint": model_name,
        "classifier": args.bgem3_classifier,
        "best_epoch": best_epoch,
        "best_val_f1_macro": val_metrics["f1_macro"],
        "model_path": str(model_save_dir),
        "classifier_path": str(classifier_path),
        "embeddings_path": str(embeddings_path),
    }
    if "loss" in val_metrics:
        extra_payload["best_val_loss"] = val_metrics["loss"]
    if test_loss is not None:
        extra_payload["test_loss"] = test_loss
    payload = save_evaluation_artifacts(
        model_dir=model_dir,
        model_name=model_key,
        test_df=splits["test"],
        text_col=args.text_col,
        y_true=split_labels["test"].tolist(),
        y_pred=y_pred,
        probs=probs,
        extra=extra_payload,
    )
    save_deployment_package(
        model_dir=model_dir,
        model_key=model_key,
        loader_type="step5_bgem3_classifier",
        framework="transformers+sklearn",
        architecture="BGEM3EmbeddingClassifier",
        text_col=args.text_col,
        max_length=args.bgem3_max_length,
        artifact_files={
            "classifier": classifier_path,
            "embeddings": embeddings_path,
            "model_info": model_save_dir / "model_info.json",
            "training_config": model_dir / "training_config.json",
            "metrics": model_dir / "reports" / "metrics.json",
        },
        base_model_name=model_name,
        model_config={
            "classifier": args.bgem3_classifier,
            "embedding_strategy": "last_hidden_state_cls_l2_normalized",
            "feature_dropout": args.bgem3_feature_dropout,
        },
        metrics=payload,
        write_hf_config=True,
    )
    logger.info("Finished BGE-M3. Test f1_macro=%.4f", payload["f1_macro"])
    return payload


def make_training_arguments(
    args: argparse.Namespace,
    output_dir: Path,
    logging_dir: Path,
    device: torch.device,
):
    from transformers import TrainingArguments

    signature = inspect.signature(TrainingArguments.__init__)
    monitor = args.transformer_monitor
    params = {
        "output_dir": str(output_dir),
        "num_train_epochs": max(args.transformer_epochs, args.transformer_min_epochs),
        "per_device_train_batch_size": args.transformer_batch_size,
        "per_device_eval_batch_size": args.transformer_batch_size,
        "learning_rate": args.transformer_lr,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": monitor,
        "greater_is_better": not monitor.endswith("loss"),
        "logging_steps": args.logging_steps,
        "logging_dir": str(logging_dir),
        "save_total_limit": args.save_total_limit,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
    }

    if "eval_strategy" in signature.parameters:
        params["eval_strategy"] = "epoch"
    else:
        params["evaluation_strategy"] = "epoch"

    if "logging_strategy" in signature.parameters:
        params["logging_strategy"] = "epoch"

    optional_params = {
        "weight_decay": args.transformer_weight_decay,
        "warmup_ratio": args.transformer_warmup_ratio,
        "label_smoothing_factor": args.transformer_label_smoothing,
        "lr_scheduler_type": args.transformer_lr_scheduler_type,
    }
    for name, value in optional_params.items():
        if name in signature.parameters:
            params[name] = value

    if "use_cpu" in signature.parameters:
        params["use_cpu"] = device.type == "cpu"
    elif "no_cuda" in signature.parameters:
        params["no_cuda"] = device.type == "cpu"

    if "fp16" in signature.parameters:
        params["fp16"] = bool(args.fp16 and device.type == "cuda")

    return TrainingArguments(**params)


def build_transformer_config(model_name: str, args: argparse.Namespace):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABEL_IDS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    dropout_values = {
        "hidden_dropout_prob": args.transformer_hidden_dropout,
        "attention_probs_dropout_prob": args.transformer_attention_dropout,
        "classifier_dropout": args.transformer_classifier_dropout,
    }
    for attr, value in dropout_values.items():
        if hasattr(config, attr):
            setattr(config, attr, value)
    return config


def run_transformer_model(
    model_key: str,
    args: argparse.Namespace,
    splits: dict[str, pd.DataFrame],
    run_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    from datasets import Dataset as HFDataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
    )

    spec = TRANSFORMER_SPECS[model_key]
    model_name = read_step4_constant(
        spec["step4_file"],
        "MODEL_NAME",
        spec["fallback_model_name"],
    )

    model_dir = run_dir / "models" / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(f"pipeline.{model_key}", model_dir / "training.log")
    logger.info("Starting %s training from checkpoint: %s", model_key, model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=spec["use_fast"])

    def tokenize_fn(batch: dict[str, list[str]]) -> dict[str, object]:
        return tokenizer(batch["text"], truncation=True, max_length=args.transformer_max_length)

    def to_hf_dataset(df: pd.DataFrame) -> HFDataset:
        base = pd.DataFrame(
            {
                "text": df[args.text_col].apply(clean_text),
                "labels": df["label_id"].astype(int),
            }
        )
        ds = HFDataset.from_pandas(base, preserve_index=False)
        return ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    train_ds = to_hf_dataset(splits["train"])
    val_ds = to_hf_dataset(splits["val"])
    test_ds = to_hf_dataset(splits["test"])

    model_config = build_transformer_config(model_name, args)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_trainer_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        return compute_metric_dict(labels, preds)

    training_args = make_training_arguments(
        args=args,
        output_dir=model_dir / "checkpoints",
        logging_dir=model_dir / "logs",
        device=device,
    )
    save_json(
        model_dir / "training_config.json",
        {
            **vars(args),
            "model_key": model_key,
            "model_name": model_name,
            "step4_file": str(spec["step4_file"]),
            "tokenizer_use_fast": spec["use_fast"],
            "dropout": {
                "hidden_dropout_prob": getattr(model_config, "hidden_dropout_prob", None),
                "attention_probs_dropout_prob": getattr(model_config, "attention_probs_dropout_prob", None),
                "classifier_dropout": getattr(model_config, "classifier_dropout", None),
            },
            "loss_function": {
                "name": "Trainer sequence-classification cross-entropy",
                "label_smoothing": args.transformer_label_smoothing,
            },
            "regularization": {
                "weight_decay": args.transformer_weight_decay,
                "warmup_ratio": args.transformer_warmup_ratio,
                "hidden_dropout_prob": getattr(model_config, "hidden_dropout_prob", None),
                "attention_probs_dropout_prob": getattr(model_config, "attention_probs_dropout_prob", None),
                "classifier_dropout": getattr(model_config, "classifier_dropout", None),
            },
        },
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": data_collator,
        "compute_metrics": compute_trainer_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if args.transformer_patience > 0 and "callbacks" in trainer_signature.parameters:
        trainer_kwargs["callbacks"] = [
            MinimumEpochEarlyStoppingCallback(
                early_stopping_patience=args.transformer_patience,
                early_stopping_threshold=args.transformer_min_delta,
                min_epochs=args.transformer_min_epochs,
                metric_name=args.transformer_monitor,
                greater_is_better=not args.transformer_monitor.endswith("loss"),
            )
        ]

    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    log_history = trainer.state.log_history
    save_json(model_dir / "logs" / "trainer_log_history.json", log_history)
    if log_history:
        write_csv(pd.DataFrame(log_history), model_dir / "logs" / "trainer_log_history.csv")
    model_label = MODEL_LABELS_VI.get(model_key, model_key)
    loss_history = transformer_epoch_loss_history(log_history)
    save_training_history_artifacts(model_dir, loss_history, f"Hàm mất mát {model_label}")

    model_save_dir = model_dir / "model"
    trainer.save_model(str(model_save_dir))
    tokenizer.save_pretrained(str(model_save_dir))

    validation_metrics = trainer.evaluate(eval_dataset=val_ds)
    save_validation_metrics_artifacts(model_dir, validation_metrics)

    test_output = trainer.predict(test_ds)
    logits = test_output.predictions[0] if isinstance(test_output.predictions, tuple) else test_output.predictions
    y_pred = np.argmax(logits, axis=-1).astype(int).tolist()
    y_true = test_output.label_ids.astype(int).tolist()
    probs = softmax_np(np.asarray(logits))

    payload = save_evaluation_artifacts(
        model_dir=model_dir,
        model_name=model_key,
        test_df=splits["test"],
        text_col=args.text_col,
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        extra={
            "model_checkpoint": model_name,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "model_path": str(model_save_dir),
        },
    )
    save_deployment_package(
        model_dir=model_dir,
        model_key=model_key,
        loader_type="huggingface_sequence_classification",
        framework="transformers",
        architecture=model.__class__.__name__,
        text_col=args.text_col,
        max_length=args.transformer_max_length,
        artifact_files={
            "model_dir": model_save_dir,
            "training_config": model_dir / "training_config.json",
            "metrics": model_dir / "reports" / "metrics.json",
            "best_checkpoint": Path(trainer.state.best_model_checkpoint)
            if trainer.state.best_model_checkpoint
            else None,
        },
        base_model_name=model_name,
        model_config={
            "tokenizer_use_fast": spec["use_fast"],
            "hidden_dropout_prob": getattr(model_config, "hidden_dropout_prob", None),
            "attention_probs_dropout_prob": getattr(model_config, "attention_probs_dropout_prob", None),
            "classifier_dropout": getattr(model_config, "classifier_dropout", None),
        },
        metrics=payload,
        write_hf_config=False,
    )
    logger.info("Finished %s. Test f1_macro=%.4f", model_key, payload["f1_macro"])
    return payload


def resolve_deployment_model_dir(model_dir: str | Path) -> Path:
    path = Path(model_dir)
    if (path / "deployment_config.json").exists():
        return path
    if (path / "model" / "deployment_config.json").exists():
        return path / "model"
    raise FileNotFoundError(
        f"Cannot find deployment_config.json in {path} or {path / 'model'}."
    )


def resolve_deployment_artifact(model_save_dir: Path, deployment_config: dict[str, object], name: str) -> Path:
    artifacts = deployment_config.get("artifacts", {})
    if not isinstance(artifacts, dict) or not artifacts.get(name):
        raise KeyError(f"Missing artifact {name!r} in deployment_config.json.")
    return (model_save_dir / str(artifacts[name])).resolve()


def predict_lstm_texts(
    model_save_dir: Path,
    deployment_config: dict[str, object],
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> tuple[list[int], np.ndarray]:
    checkpoint_path = resolve_deployment_artifact(model_save_dir, deployment_config, "weights")
    vocab_path = resolve_deployment_artifact(model_save_dir, deployment_config, "vocab")
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    vocab = load_json(vocab_path)
    config = checkpoint.get("model_config") or load_json(model_save_dir / "config.json").get("step5_model_config", {})
    model = BiLSTMSentimentClassifier(
        vocab_size=int(config["vocab_size"]),
        embedding_dim=int(config["embedding_dim"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        num_classes=int(config["num_classes"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    max_length = int(config["max_length"])
    preds: list[int] = []
    probs: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = [encode_lstm_text(text, vocab, max_length) for text in batch_texts]
            input_ids = torch.tensor([item[0] for item in encoded], dtype=torch.long, device=device)
            lengths = torch.tensor([item[1] for item in encoded], dtype=torch.long, device=device)
            logits = model(input_ids, lengths)
            batch_probs = torch.softmax(logits, dim=1)
            preds.extend(torch.argmax(batch_probs, dim=1).detach().cpu().tolist())
            probs.extend(batch_probs.detach().cpu().tolist())
    return preds, np.asarray(probs, dtype=np.float32)


def predict_bgem3_texts(
    model_save_dir: Path,
    deployment_config: dict[str, object],
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> tuple[list[int], np.ndarray | None]:
    import joblib
    from transformers import AutoModel, AutoTokenizer

    classifier_path = resolve_deployment_artifact(model_save_dir, deployment_config, "classifier")
    config = load_json(model_save_dir / "config.json")
    base_model_name = config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("BGE-M3 deployment config is missing base_model_name_or_path.")

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_name))
    encoder = AutoModel.from_pretrained(str(base_model_name)).to(device)
    encoder.eval()
    max_length = int(config.get("max_length", 512))
    embeddings = extract_bgem3_embeddings(
        texts=texts,
        tokenizer=tokenizer,
        model=encoder,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        desc="BGE-M3 predict",
    )
    classifier = joblib.load(classifier_path)
    preds = classifier.predict(embeddings).astype(int).tolist()
    probs = classifier_probabilities(classifier, embeddings)
    return preds, probs


def predict_transformer_texts(
    model_save_dir: Path,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> tuple[list[int], np.ndarray]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_save_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_save_dir)).to(device)
    model.eval()
    preprocessor = load_json(model_save_dir / "preprocessor_config.json")
    tokenizer_config = preprocessor.get("tokenizer", {}) if isinstance(preprocessor, dict) else {}
    max_length = int(tokenizer_config.get("max_length", 256))

    preds: list[int] = []
    probs: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = [clean_text(text) for text in texts[start : start + batch_size]]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            logits = model(**inputs).logits
            batch_probs = torch.softmax(logits, dim=1)
            preds.extend(torch.argmax(batch_probs, dim=1).detach().cpu().tolist())
            probs.extend(batch_probs.detach().cpu().tolist())
    return preds, np.asarray(probs, dtype=np.float32)


def prediction_dataframe(texts: Sequence[str], preds: Sequence[int], probs: np.ndarray | None) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "text": list(texts),
            "pred_label_id": [int(label_id) for label_id in preds],
            "pred_label_name": [ID2LABEL[int(label_id)] for label_id in preds],
        }
    )
    if probs is not None:
        for label_id in LABEL_IDS:
            df[f"prob_{ID2LABEL[label_id]}"] = probs[:, label_id]
    return df


def load_prediction_inputs(args: argparse.Namespace) -> tuple[list[str], pd.DataFrame | None]:
    if args.input_file:
        input_df = read_table(args.input_file)
        if args.predict_text_col not in input_df.columns:
            raise ValueError(
                f"Missing --predict_text_col {args.predict_text_col!r}. "
                f"Available columns: {list(input_df.columns)}"
            )
        texts = input_df[args.predict_text_col].apply(clean_text).tolist()
        return texts, input_df
    if args.input_text:
        return [clean_text(text) for text in args.input_text], None
    raise ValueError("Predict mode needs --input_text or --input_file.")


def run_predict_mode(args: argparse.Namespace, device: torch.device) -> None:
    model_save_dir = resolve_deployment_model_dir(args.model_dir)
    deployment_config = load_json(model_save_dir / "deployment_config.json")
    if not isinstance(deployment_config, dict):
        raise ValueError("deployment_config.json must contain a JSON object.")
    texts, input_df = load_prediction_inputs(args)
    loader_type = str(deployment_config.get("loader_type"))

    if loader_type == "step5_lstm":
        preds, probs = predict_lstm_texts(model_save_dir, deployment_config, texts, device, args.predict_batch_size)
    elif loader_type == "step5_bgem3_classifier":
        preds, probs = predict_bgem3_texts(model_save_dir, deployment_config, texts, device, args.predict_batch_size)
    elif loader_type == "huggingface_sequence_classification":
        preds, probs = predict_transformer_texts(model_save_dir, texts, device, args.predict_batch_size)
    else:
        raise ValueError(f"Unsupported loader_type {loader_type!r}.")

    pred_df = prediction_dataframe(texts, preds, probs)
    if input_df is not None:
        pred_df = pd.concat([input_df.reset_index(drop=True), pred_df.drop(columns=["text"])], axis=1)

    if args.output_file:
        write_csv(pred_df, Path(args.output_file))
    else:
        print(pred_df.to_string(index=False))


def save_run_deployment_manifest(run_dir: Path, metrics_rows: list[dict[str, object]]) -> None:
    models = []
    for row in metrics_rows:
        model_key = str(row.get("model", ""))
        if not model_key:
            continue
        model_root = run_dir / "models" / model_key
        models.append(
            {
                "model": model_key,
                "load_dir": str(model_root / "model"),
                "deployment_config": str(model_root / "model" / "deployment_config.json"),
                "reports_dir": str(model_root / "reports"),
                "logs_dir": str(model_root / "logs"),
                "f1_macro": row.get("f1_macro"),
                "accuracy": row.get("accuracy"),
            }
        )
    save_json(
        run_dir / "deployment_manifest.json",
        {
            "format": "step5-run-deployment-manifest",
            "format_version": 1,
            "models": models,
        },
    )


def resolve_requested_models(raw_models: Sequence[str]) -> list[str]:
    normalized = [model.lower() for model in raw_models]
    if "all" in normalized:
        return list(ALL_MODEL_KEYS)
    result = []
    for model in normalized:
        if model not in ALL_MODEL_KEYS:
            valid = "/".join(("all", *ALL_MODEL_KEYS))
            raise ValueError(f"Unknown model {model!r}. Use {valid}.")
        if model not in result:
            result.append(model)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM, PhoBERT, SimCSE, and BGE-M3 sentiment models.")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--models", nargs="+", default=["all"], help="all, lstm, phobert, simcse, bgem3")
    parser.add_argument("--text_col", default="word_segmented")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument(
        "--split_strategy",
        choices=["label", "label_data_type", "label_data_type_emotion"],
        default="label_data_type",
        help="How to stratify train/val/test splits.",
    )
    parser.add_argument("--rebuild_data", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--model_dir", default=None, help="Model folder for --mode predict.")
    parser.add_argument("--input_text", nargs="*", default=None, help="One or more texts for --mode predict.")
    parser.add_argument("--input_file", default=None, help="CSV/XLSX file for --mode predict.")
    parser.add_argument("--predict_text_col", default="text", help="Text column in --input_file.")
    parser.add_argument("--output_file", default=None, help="CSV output path for --mode predict.")
    parser.add_argument("--predict_batch_size", type=int, default=32)

    parser.add_argument("--lstm_epochs", type=int, default=50)
    parser.add_argument("--lstm_batch_size", type=int, default=64)
    parser.add_argument("--lstm_lr", type=float, default=5e-5)
    parser.add_argument("--lstm_embedding_dim", type=int, default=128)
    parser.add_argument("--lstm_hidden_size", type=int, default=128)
    parser.add_argument("--lstm_num_layers", type=int, default=1)
    parser.add_argument("--lstm_dropout", type=float, default=0.5)
    parser.add_argument("--lstm_max_length", type=int, default=128)
    parser.add_argument("--lstm_max_vocab", type=int, default=30000)
    parser.add_argument("--lstm_min_freq", type=int, default=2)
    parser.add_argument("--lstm_patience", type=int, default=5)
    parser.add_argument("--lstm_min_epochs", type=int, default=20)
    parser.add_argument("--lstm_monitor", choices=["val_loss", "val_unweighted_loss", "val_f1_macro"], default="val_loss")
    parser.add_argument("--lstm_min_delta", type=float, default=1e-4)
    parser.add_argument("--lstm_weight_decay", type=float, default=1e-2)
    parser.add_argument("--lstm_label_smoothing", type=float, default=0.05)
    parser.add_argument("--lstm_grad_clip", type=float, default=1.0)
    parser.add_argument("--no_class_weight", action="store_true")

    parser.add_argument("--transformer_epochs", type=int, default=50)
    parser.add_argument("--transformer_batch_size", type=int, default=16)
    # Giảm LR để bớt overfit nhanh
    parser.add_argument("--transformer_lr", type=float, default=1e-5)
    parser.add_argument("--transformer_max_length", type=int, default=256)
    # Cho model có thêm cơ hội, nhưng không kéo quá lâu
    parser.add_argument("--transformer_patience", type=int, default=5)
    # Khuyên dùng 5 thay vì 20 nếu mục tiêu là model tốt nhất
    parser.add_argument("--transformer_min_epochs", type=int, default=5)
    # Nếu bài toán chấm F1 macro thì monitor F1 macro hợp lý hơn loss
    parser.add_argument("--transformer_monitor", choices=["eval_loss", "eval_f1_macro"], default="eval_f1_macro")
    parser.add_argument("--transformer_min_delta", type=float, default=1e-4)
    # Tăng regularization
    parser.add_argument("--transformer_weight_decay", type=float, default=0.05)
    parser.add_argument("--transformer_label_smoothing", type=float, default=0.1)
    parser.add_argument("--transformer_warmup_ratio", type=float, default=0.1)
    parser.add_argument("--transformer_lr_scheduler_type", default="linear")
    # Tăng dropout
    parser.add_argument("--transformer_hidden_dropout", type=float, default=0.3)
    parser.add_argument("--transformer_attention_dropout", type=float, default=0.3)
    parser.add_argument("--transformer_classifier_dropout", type=float, default=0.4)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--bgem3_classifier", choices=["logreg", "mlp"], default="mlp")
    parser.add_argument("--bgem3_epochs", type=int, default=50)
    parser.add_argument("--bgem3_patience", type=int, default=5)
    parser.add_argument("--bgem3_min_epochs", type=int, default=20)
    parser.add_argument("--bgem3_min_delta", type=float, default=1e-4)
    parser.add_argument("--bgem3_batch_size", type=int, default=16)
    parser.add_argument("--bgem3_max_length", type=int, default=512)
    parser.add_argument("--bgem3_max_iter", type=int, default=None)
    parser.add_argument("--bgem3_logreg_c", type=float, default=0.5)
    parser.add_argument("--bgem3_mlp_hidden_size", type=int, default=256)
    parser.add_argument("--bgem3_mlp_alpha", type=float, default=1e-3)
    parser.add_argument("--bgem3_feature_dropout", type=float, default=0.1)
    parser.add_argument("--bgem3_validation_fraction", type=float, default=0.15)
    args = parser.parse_args()
    args.lstm_min_epochs = max(1, args.lstm_min_epochs)
    args.transformer_min_epochs = max(1, args.transformer_min_epochs)
    args.bgem3_min_epochs = max(1, args.bgem3_min_epochs)
    args.lstm_epochs = max(args.lstm_epochs, args.lstm_min_epochs)
    args.transformer_epochs = max(args.transformer_epochs, args.transformer_min_epochs)
    args.bgem3_epochs = max(args.bgem3_epochs, args.bgem3_min_epochs)
    for name in ("lstm_label_smoothing", "transformer_label_smoothing"):
        value = getattr(args, name)
        if value < 0 or value >= 1:
            parser.error(f"--{name} must be in [0, 1).")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.mode == "predict":
        if not args.model_dir:
            raise ValueError("--mode predict requires --model_dir.")
        run_predict_mode(args, device)
        return

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger("pipeline", run_dir / "pipeline.log")
    requested_models = resolve_requested_models(args.models)

    logger.info("Run directory: %s", run_dir)
    logger.info("Models: %s", ", ".join(requested_models))
    logger.info("Device: %s", device)
    save_json(run_dir / "run_config.json", vars(args))

    splits = get_or_create_data_splits(
        dataset_path=Path(args.dataset),
        data_dir=Path(args.data_dir),
        text_col=args.text_col,
        label_col=args.label_col,
        val_size=args.val_size,
        test_size=args.test_size,
        split_strategy=args.split_strategy,
        seed=args.seed,
        rebuild_data=args.rebuild_data,
        logger=logger,
    )

    metrics_rows = []
    for model_key in requested_models:
        try:
            if model_key == "lstm":
                payload = run_lstm_model(args, splits, run_dir, device)
            elif model_key == "bgem3":
                payload = run_bgem3_model(args, splits, run_dir, device)
            else:
                payload = run_transformer_model(model_key, args, splits, run_dir, device)
            metrics_rows.append({k: v for k, v in payload.items() if k != "classification_report"})
        except Exception as exc:
            logger.exception("Model %s failed: %s", model_key, exc)
            if not args.continue_on_error:
                raise

    if metrics_rows:
        comparison_dir = run_dir / "reports"
        metrics_df = pd.DataFrame(metrics_rows)
        write_csv(metrics_df, comparison_dir / "model_comparison_metrics.csv")
        save_json(comparison_dir / "model_comparison_metrics.json", metrics_rows)
        plot_model_comparison(metrics_df, comparison_dir / "model_comparison_metrics.png")
        save_run_deployment_manifest(run_dir, metrics_rows)

    logger.info("Pipeline finished. Outputs saved under: %s", run_dir)


if __name__ == "__main__":
    main()
