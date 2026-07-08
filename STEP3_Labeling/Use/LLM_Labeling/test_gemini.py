# -*- coding: utf-8 -*-
"""
Quick smoke test for Gemini sentiment labeling.

Run:
    python STEP3_Labeling/LLM_Labeling/test_gemini.py

Optional:
    python STEP3_Labeling/LLM_Labeling/test_gemini.py --model gemini-3.5-flash
"""

from __future__ import annotations

import argparse
import json
import time

from label_gemini import MODEL_NAME, build_client, predict_one


TEST_SENTENCES = [
    "Hôm nay mình rất vui vì bài thuyết trình được cô khen và nhóm làm việc rất tốt.",
    "Mình thấy môn này bình thường, không quá thích cũng không quá ghét.",
    "Mình rất thất vọng vì điểm thi thấp dù đã cố gắng học rất nhiều.",
    "Cảm ơn mọi người đã hỗ trợ, mình cảm thấy nhẹ nhõm hơn nhiều.",
    "Lịch học dày quá, deadline liên tục làm mình kiệt sức và chán nản.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Gemini sentiment on a few Vietnamese sentences.")
    parser.add_argument("--model", default=MODEL_NAME, help="Gemini model name")
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=3.0)
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = build_client()

    print(f"Model: {args.model}")
    print()

    for index, text in enumerate(TEST_SENTENCES, start=1):
        prediction = predict_one(
            client=client,
            text=text,
            model_name=args.model,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )

        print(f"#{index}")
        print(f"Text       : {text}")
        print(f"Label      : {prediction['sentiment_label']}")
        print(f"Score      : {prediction['sentiment_score']:.6f}")
        print(f"Raw label  : {prediction['sentiment_raw_label']}")
        print("Raw scores :")
        print(json.dumps(json.loads(prediction["sentiment_raw_scores"]), ensure_ascii=False, indent=2))
        print()

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
