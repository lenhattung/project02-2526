from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from threading import Lock
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models import AIAnalysisResult, NewsItem


LABEL_PRIORITY = ("gemini", "phobert", "simcse", "bgem3")
LABEL_TEXT = {0: "negative", 1: "neutral", 2: "positive"}
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = Lock()


def sentiment_to_code(label: str) -> int:
    cleaned = (label or "").strip().lower()
    mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
        "tieu_cuc": 0,
        "trung_lap": 1,
        "tich_cuc": 2,
        "0": 0,
        "1": 1,
        "2": 2,
    }
    return mapping.get(cleaned, 1)


def code_to_sentiment(code: int | None) -> str:
    if code is None:
        return "unknown"
    return LABEL_TEXT.get(int(code), "neutral")


def normalize_model_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        code = int(value)
        return code if code in (0, 1, 2) else None
    text = str(value).strip().lower()
    if not text:
        return None
    mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
        "tieu_cuc": 0,
        "trung_lap": 1,
        "tich_cuc": 2,
        "0": 0,
        "1": 1,
        "2": 2,
    }
    return mapping.get(text)


def maybe_segment_text(text: str) -> str:
    try:
        from underthesea import word_tokenize  # type: ignore

        return word_tokenize(text, format="text") or text
    except Exception:
        return text


@dataclass
class AIAnalysis:
    provider: str
    model_name: str
    summary: str
    category: str
    importance_score: float
    attention_required: bool
    suggested_action: str
    sentiment_label: str
    sentiment_code: int
    status_label: str
    raw_result_json: dict[str, Any]


@dataclass
class MultiModelLabels:
    gemini_label: int | None
    simcse_label: int | None
    phobert_label: int | None
    bgem3_label: int | None
    voted_label: int | None
    label_status: str
    errors: dict[str, str]


class MockAIProvider:
    provider = "mock"
    model_name = "rule-based-demo"

    def analyze(self, content: str) -> AIAnalysis:
        lowered = content.lower()
        urgent_words = ["khung hoang", "canh bao", "bao luc", "khieu nai", "gap", "nguy hiem", "lua dao"]
        academic_words = ["lich hoc", "diem", "thi", "hoc phi", "phong hoc"]
        activity_words = ["tinh nguyen", "su kien", "ngay hoi", "clb", "hoat dong"]

        attention = any(word in lowered for word in urgent_words)
        if any(word in lowered for word in academic_words):
            category = "hoc_tap"
        elif any(word in lowered for word in activity_words):
            category = "hoat_dong"
        elif attention:
            category = "canh_bao"
        else:
            category = "doi_song_sinh_vien"

        negative = any(word in lowered for word in ["khong hai long", "te", "loi", "that vong", "buc xuc"])
        positive = any(word in lowered for word in ["tot", "cam on", "hay", "tich cuc"])
        sentiment = "negative" if negative else "positive" if positive else "neutral"
        sentiment_code = sentiment_to_code(sentiment)
        score = 0.88 if attention else 0.62 if negative else 0.38 if positive else 0.5
        summary = content.strip().replace("\n", " ")[:220]
        if len(content) > 220:
            summary += "..."

        return AIAnalysis(
            provider=self.provider,
            model_name=self.model_name,
            summary=summary or "Khong co noi dung de tom tat.",
            category=category,
            importance_score=score,
            attention_required=attention or score >= 0.8,
            suggested_action="Can bo CTSV can xem va phan cong xu ly som." if attention else "Theo doi va cap nhat trang thai khi can.",
            sentiment_label=sentiment,
            sentiment_code=sentiment_code,
            status_label="needs_attention" if attention else "analyzed",
            raw_result_json={"provider": self.provider, "rules": {"attention": attention, "sentiment": sentiment, "sentiment_code": sentiment_code}},
        )


class GeminiAIProvider:
    provider = "gemini"

    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    def _build_prompt(self, content: str) -> str:
        return (
            "Ban la he thong gan nhan du lieu tin bai viet CTSV. "
            "Hay phan tich noi dung va tra ve duy nhat 1 JSON object voi cac khoa: "
            "summary, category, importance_score, attention_required, suggested_action, "
            "sentiment_label, sentiment_code, status_label. "
            "sentiment_label chi duoc la negative, neutral hoac positive. "
            "sentiment_code phai la 0 neu negative, 1 neu neutral, 2 neu positive. "
            "importance_score la so thuc tu 0 den 1. "
            "status_label nen la needs_attention neu can chu y gap, nguoc lai la analyzed. "
            "category nen ngan gon kieu snake_case. "
            "Khong them markdown, khong them giai thich ngoai JSON.\n\n"
            f"Noi dung:\n{content}"
        )

    def _extract_json(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        candidate = match.group(0) if match else cleaned
        data = json.loads(candidate)
        if not isinstance(data, dict):
            raise ValueError("Gemini response is not a JSON object")
        return data

    def analyze(self, content: str) -> AIAnalysis:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": self._build_prompt(content)}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        }
        with httpx.Client(timeout=45) as client:
            response = client.post(url, params={"key": self.api_key}, json=payload)
            response.raise_for_status()
            body = response.json()
        text = body["candidates"][0]["content"]["parts"][0]["text"]
        data = self._extract_json(text)
        sentiment_label = str(data.get("sentiment_label", "neutral")).lower()
        sentiment_code = int(data.get("sentiment_code", sentiment_to_code(sentiment_label)))
        if sentiment_code not in (0, 1, 2):
            sentiment_code = sentiment_to_code(sentiment_label)
        return AIAnalysis(
            provider=self.provider,
            model_name=self.model_name,
            summary=str(data.get("summary") or content[:220]),
            category=str(data.get("category") or "chua_phan_loai"),
            importance_score=max(0.0, min(1.0, float(data.get("importance_score", 0.5)))),
            attention_required=bool(data.get("attention_required", False)),
            suggested_action=str(data.get("suggested_action") or "Theo doi va cap nhat trang thai khi can."),
            sentiment_label=sentiment_label,
            sentiment_code=sentiment_code,
            status_label=str(data.get("status_label") or ("needs_attention" if data.get("attention_required") else "analyzed")),
            raw_result_json={"provider": self.provider, "response": body, "parsed": data},
        )


class LocalSequenceClassifier:
    def __init__(self, model_key: str, model_dir: Path):
        self.model_key = model_key
        self.model_dir = model_dir
        self._tokenizer = None
        self._model = None

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir), local_files_only=True)
        self._model.eval()
        return self._tokenizer, self._model

    def predict(self, content: str) -> int:
        tokenizer, model = self._load()
        text = maybe_segment_text(content) if self.model_key in {"phobert", "simcse"} else content
        encoded = tokenizer(text, truncation=True, padding=True, return_tensors="pt")

        import torch  # type: ignore

        with torch.no_grad():
            logits = model(**encoded).logits
            prediction = int(torch.argmax(logits, dim=-1).item())
        normalized = normalize_model_label(prediction)
        if normalized is None:
            raise ValueError(f"{self.model_key} returned unsupported label: {prediction}")
        return normalized


class BGEM3Classifier:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._tokenizer = None
        self._model = None
        self._classifier = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._tokenizer is not None and self._model is not None and self._classifier is not None:
            return self._tokenizer, self._model, self._classifier

        from joblib import load  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        classifier_path = self.model_dir / "classifier_mlp.joblib"
        if not classifier_path.exists():
            raise FileNotFoundError(f"Missing BGE-M3 classifier at {classifier_path}")

        # The exported folder only contains the classifier. The base BGE-M3 weights
        # must be available in local cache or mounted alongside the container.
        self._tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", local_files_only=True)
        self._model = AutoModel.from_pretrained("BAAI/bge-m3", local_files_only=True)
        self._model.eval()
        self._classifier = load(classifier_path)
        return self._tokenizer, self._model, self._classifier

    def predict(self, content: str) -> int:
        tokenizer, model, classifier = self._load()

        import numpy as np  # type: ignore
        import torch  # type: ignore

        encoded = tokenizer(content, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1)
            hidden = outputs.last_hidden_state
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        prediction = classifier.predict(np.asarray(pooled.cpu()))[0]
        normalized = normalize_model_label(prediction)
        if normalized is None:
            raise ValueError(f"bgem3 returned unsupported label: {prediction}")
        return normalized


def get_ai_provider():
    settings = get_settings()
    if settings.ai_provider.lower() == "gemini" and settings.gemini_api_key:
        return GeminiAIProvider(settings.gemini_api_key, settings.gemini_model)
    return MockAIProvider()


def analyze_with_fallback(content: str) -> AIAnalysis:
    provider = get_ai_provider()
    try:
        return provider.analyze(content)
    except Exception as exc:
        fallback = MockAIProvider().analyze(content)
        fallback.raw_result_json = {
            "provider": fallback.provider,
            "fallback_from": getattr(provider, "provider", "unknown"),
            "fallback_reason": str(exc),
        }
        return fallback


def get_labeler(model_key: str):
    settings = get_settings()
    model_root = Path(settings.local_models_dir)
    cache_key = f"{model_key}:{model_root}"
    with _MODEL_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        if model_key in {"phobert", "simcse"}:
            labeler = LocalSequenceClassifier(model_key, model_root / model_key / "model")
        elif model_key == "bgem3":
            labeler = BGEM3Classifier(model_root / "bgem3" / "model")
        else:
            raise ValueError(f"Unsupported model key: {model_key}")

        _MODEL_CACHE[cache_key] = labeler
        return labeler


def predict_local_label(model_key: str, content: str) -> int:
    labeler = get_labeler(model_key)
    return labeler.predict(content)


def compute_voted_label(labels: dict[str, int | None]) -> int | None:
    usable = {name: code for name, code in labels.items() if code is not None}
    if not usable:
        return None

    counts = Counter(usable.values())
    highest = max(counts.values())
    winners = {code for code, count in counts.items() if count == highest}
    if len(winners) == 1:
        return next(iter(winners))

    for source in LABEL_PRIORITY:
        code = usable.get(source)
        if code in winners:
            return code
    return next(iter(winners))


def collect_model_labels(content: str, gemini_code: int | None = None) -> MultiModelLabels:
    settings = get_settings()
    errors: dict[str, str] = {}
    labels: dict[str, int | None] = {
        "gemini": gemini_code,
        "simcse": None,
        "phobert": None,
        "bgem3": None,
    }

    if gemini_code is None:
        errors["gemini"] = "Gemini label unavailable"

    if settings.enable_local_models:
        for model_key in ("simcse", "phobert", "bgem3"):
            try:
                labels[model_key] = predict_local_label(model_key, content)
            except Exception as exc:
                errors[model_key] = str(exc)
    else:
        for model_key in ("simcse", "phobert", "bgem3"):
            errors[model_key] = "Local models are disabled"

    voted = compute_voted_label(labels)
    available_count = sum(1 for value in labels.values() if value is not None)
    if available_count == len(labels):
        status = "complete"
    elif available_count > 0:
        status = "partial"
    else:
        status = "failed"

    return MultiModelLabels(
        gemini_label=labels["gemini"],
        simcse_label=labels["simcse"],
        phobert_label=labels["phobert"],
        bgem3_label=labels["bgem3"],
        voted_label=voted,
        label_status=status,
        errors=errors,
    )


def upsert_ai_result(db: Session, item: NewsItem, result: AIAnalysis) -> AIAnalysisResult:
    existing = (
        db.query(AIAnalysisResult)
        .filter(
            AIAnalysisResult.news_item_id == item.id,
            AIAnalysisResult.provider == result.provider,
            AIAnalysisResult.model_name == result.model_name,
        )
        .first()
    )
    record = existing or AIAnalysisResult(news_item_id=item.id, provider=result.provider, model_name=result.model_name)
    record.summary = result.summary
    record.category = result.category
    record.importance_score = result.importance_score
    record.attention_required = result.attention_required
    record.suggested_action = result.suggested_action
    record.sentiment_label = result.sentiment_label
    record.sentiment_code = result.sentiment_code
    record.status_label = result.status_label
    record.raw_result_json = result.raw_result_json
    db.add(record)
    db.flush()
    return record


def persist_analysis(db: Session, item: NewsItem) -> AIAnalysisResult:
    result = analyze_with_fallback(item.content)
    record = upsert_ai_result(db, item, result)

    gemini_code = result.sentiment_code if result.provider == "gemini" else None
    labels = collect_model_labels(item.content, gemini_code=gemini_code)

    item.topic = result.category
    item.importance_level = "high" if result.importance_score >= 0.8 else "normal"
    item.status = result.status_label
    item.gemini_label = labels.gemini_label
    item.simcse_label = labels.simcse_label
    item.phobert_label = labels.phobert_label
    item.bgem3_label = labels.bgem3_label
    item.voted_label = labels.voted_label
    item.label_status = labels.label_status
    item.label_error_json = labels.errors or None
    item.labeled_at = datetime.utcnow()
    db.add(item)
    db.flush()
    return record


def analyze_news_item_by_id(news_id: int) -> bool:
    db = SessionLocal()
    try:
        item = db.get(NewsItem, news_id)
        if not item:
            return False
        persist_analysis(db, item)
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def analyze_news_items_by_id(news_ids: list[int]) -> int:
    processed = 0
    for news_id in news_ids:
        if analyze_news_item_by_id(news_id):
            processed += 1
    return processed
