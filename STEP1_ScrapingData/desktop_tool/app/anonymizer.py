from __future__ import annotations

from contextlib import redirect_stdout
from functools import lru_cache
import importlib.util
import io
from pathlib import Path
import sys
import types


ROOT_DIR = Path(__file__).resolve().parents[3]
ANONYMIZE_PATH = ROOT_DIR / "STEP2_Anonymize" / "Anonymize_CRF.py"


@lru_cache(maxsize=1)
def _load_pipeline():
    if not ANONYMIZE_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file anonymize: {ANONYMIZE_PATH}")
    spec = importlib.util.spec_from_file_location("ctsv_anonymize_crf", ANONYMIZE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không load được module anonymize từ {ANONYMIZE_PATH}")
    module = importlib.util.module_from_spec(spec)
    injected_modules: dict[str, types.ModuleType] = {}

    def ensure_stub(name: str, factory):
        if name in sys.modules:
            return
        stub = factory()
        injected_modules[name] = stub
        sys.modules[name] = stub

    ensure_stub("pandas", lambda: types.ModuleType("pandas"))

    def tqdm_factory():
        stub = types.ModuleType("tqdm")

        class TqdmStub:
            @staticmethod
            def pandas():
                return None

        stub.tqdm = TqdmStub
        return stub

    ensure_stub("tqdm", tqdm_factory)

    def underthesea_factory():
        stub = types.ModuleType("underthesea")
        stub.ner = lambda text: []
        return stub

    ensure_stub("underthesea", underthesea_factory)
    with redirect_stdout(io.StringIO()):
        try:
            spec.loader.exec_module(module)
        finally:
            for name in injected_modules:
                sys.modules.pop(name, None)
    if not hasattr(module, "anonymize_pipeline"):
        raise RuntimeError("Module anonymize không có hàm anonymize_pipeline")
    return module.anonymize_pipeline


def anonymize_content(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    try:
        pipeline = _load_pipeline()
        return pipeline(text)
    except Exception:
        return text
