"""Benchmark and test runner for AMD Mini-Challenge 2 (OCR).

Evaluates starter/app/app.py against sample-images and sample-labels.json
using the official scoring normalization logic.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import time

base_dir = Path(__file__).resolve().parent
app_dir = base_dir / "starter" / "app"
sys.path.insert(0, str(app_dir))

from app import recognize_image


def normalize(text: str) -> str:
    """Official Match 2 normalization:
    - Converted to uppercase
    - All whitespace removed
    - The characters - . · _ removed
    """
    t = text.upper()
    for ch in (" ", "\t", "\n", "\r", "-", ".", "·", "_"):
        t = t.replace(ch, "")
    return t


def main():
    images_dir = base_dir / "sample-images"
    labels_file = base_dir / "sample-labels.json"

    with open(labels_file, "r", encoding="utf-8") as f:
        labels = json.load(f)

    with tempfile.TemporaryDirectory() as temp_out:
        os.environ["MC2_OUTPUT_DIR"] = temp_out

        print("=== Evaluating Mini-Challenge 2 (OCR) on 10 Sample Images ===\n")
        total_score = 0
        max_score = len(labels) * 20

        for item in labels:
            n = item["n"]
            fname = item["file"]
            img_path = images_dir / fname
            expected_text = item["expected_text"]

            t0 = time.time()
            pred_text, confidence = recognize_image(img_path)
            elapsed = time.time() - t0

            norm_pred = normalize(pred_text)
            norm_exp = normalize(expected_text)

            passed = (norm_pred == norm_exp)
            score = 20 if passed else 0
            total_score += score

            status = "PASS (20/20)" if passed else f"FAIL (0/20) [got '{pred_text}', expected '{expected_text}']"
            print(f"[{n:02d}] {fname:38} -> {status} (time: {elapsed:.3f}s)")
            print(f"     Normalized: pred='{norm_pred}' expected='{norm_exp}'")

        print("\n==========================================")
        print(f"FINAL SCORE: {total_score} / {max_score}")
        print("==========================================")
        return 0 if total_score == max_score else 1


if __name__ == "__main__":
    sys.exit(main())
