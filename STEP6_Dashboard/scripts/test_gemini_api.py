from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import httpx


PROJECT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_DIR / ".env"
DEFAULT_MODEL = "gemini-1.5-flash"
DEFAULT_PROMPT = "Tra loi duy nhat mot dong: api_key_ok"


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        values[key] = value
    return values


def extract_text(body: dict) -> str:
    candidates = body.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    return "\n".join(text for text in texts if text).strip()


def list_models(api_key: str, timeout: float) -> tuple[int, dict]:
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url, params={"key": api_key})
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = {"raw_text": response.text}
        return response.status_code, body


def supported_generate_models(body: dict) -> list[str]:
    result: list[str] = []
    for model in body.get("models", []) or []:
        if not isinstance(model, dict):
            continue
        methods = model.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue
        name = str(model.get("name") or "").strip()
        if name:
            result.append(name)
    return result


def test_gemini(api_key: str, model: str, prompt: str, timeout: float) -> tuple[int, str, dict]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0},
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, params={"key": api_key}, json=payload)
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = {"raw_text": response.text}
        return response.status_code, extract_text(body), body


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Gemini API key from STEP6_Dashboard/.env")
    parser.add_argument("--env", default=str(ENV_PATH), help="Path to .env file")
    parser.add_argument("--model", default=None, help="Override Gemini model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    parser.add_argument("--list-models", action="store_true", help="List models available for this API key")
    args = parser.parse_args()

    env_values = read_env_file(Path(args.env))
    api_key = env_values.get("GEMINI_API_KEY", "").strip()
    model = (args.model or env_values.get("GEMINI_MODEL") or DEFAULT_MODEL).strip()

    if not api_key:
        print(f"ERROR: GEMINI_API_KEY not found in {args.env}")
        return 1

    print(f"Using env: {args.env}")
    print(f"API key present: yes (length={len(api_key)})")

    if args.list_models:
        try:
            status_code, body = list_models(api_key=api_key, timeout=args.timeout)
        except Exception as exc:
            print(f"REQUEST FAILED: {exc}")
            return 2
        print(f"HTTP status: {status_code}")
        models = supported_generate_models(body)
        if models:
            print("Models supporting generateContent:")
            for name in models:
                print(name)
            return 0
        print("Response body:")
        print(json.dumps(body, ensure_ascii=False, indent=2)[:4000])
        return 3

    print(f"Using model: {model}")

    try:
        status_code, text, body = test_gemini(api_key=api_key, model=model, prompt=args.prompt, timeout=args.timeout)
    except Exception as exc:
        print(f"REQUEST FAILED: {exc}")
        return 2

    print(f"HTTP status: {status_code}")
    if text:
        print("Response text:")
        print(text)
    else:
        print("Response body:")
        print(json.dumps(body, ensure_ascii=False, indent=2)[:4000])

    if 200 <= status_code < 300:
        print("RESULT: Gemini API key appears to be working.")
        return 0

    if status_code == 404:
        print("Model may be unavailable for this API key or API version.")
        try:
            list_status, list_body = list_models(api_key=api_key, timeout=args.timeout)
            print(f"ListModels HTTP status: {list_status}")
            models = supported_generate_models(list_body)
            if models:
                print("Models supporting generateContent:")
                for name in models:
                    print(name)
            else:
                print(json.dumps(list_body, ensure_ascii=False, indent=2)[:4000])
        except Exception as exc:
            print(f"ListModels failed: {exc}")

    print("RESULT: Gemini API key test failed.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
