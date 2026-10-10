"""Direct end-to-end evaluation script for AMD Mini-Challenge 3.

Imports and evaluates starter/app/app.py against mc3-corpus and sample-questions.json,
simulating the official grading harness scoring logic.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import time

# Ensure starter/app is in Python path
base_dir = Path(__file__).resolve().parent
app_dir = base_dir / "starter" / "app"
sys.path.insert(0, str(app_dir))

from app import index, answer


def normalize_answer(val: str) -> str:
    # Match 2 & 3 normalization: uppercased, whitespace removed, - . · _ removed
    v = val.upper()
    for ch in (" ", "\t", "\n", "-", ".", "·", "_", "°", "C", "DEG", "DEGREES"):
        v = v.replace(ch, "")
    return v


def main():
    corpus_dir = base_dir / "mc3-corpus"
    questions_file = base_dir / "sample-questions.json"

    with open(questions_file, "r", encoding="utf-8") as f:
        bench = json.load(f)

    with tempfile.TemporaryDirectory() as temp_out:
        os.environ["MC2_OUTPUT_DIR"] = temp_out

        print("=== Step 1: Running index() ===")
        t0 = time.time()
        index(corpus_dir)
        print(f"Index completed in {time.time() - t0:.2f}s\n")

        print("=== Step 2: Evaluating 10 Queries ===")
        total_score = 0
        max_score = len(bench["queries"]) * 20

        for q in bench["queries"]:
            q_num = q["n"]
            query_id = f"query_{q_num:02d}"
            query_text = q["query"]
            expected_ans = q["expected_answer"]
            expected_cits = set(q["expected_citations"])
            aliases = [expected_ans] + q.get("answer_aliases", [])

            t_start = time.time()
            ans, cits_list, confidence = answer(corpus_dir, query_text)
            elapsed = time.time() - t_start

            cits = set(cits_list)

            # Match answer against expected or aliases using normalized comparison
            norm_ans = normalize_answer(ans)
            norm_aliases = [normalize_answer(a) for a in aliases]
            ans_ok = (norm_ans in norm_aliases) if expected_ans else (ans == "")

            # Citations must match EXACT set
            cits_ok = (cits == expected_cits)

            if ans_ok and cits_ok:
                score = 20
                status = "PASS (20/20)"
            else:
                score = 0
                status = "FAIL (0/20)"
                if not ans_ok:
                    status += f" [Wrong Answer: got '{ans}', expected '{expected_ans}']"
                if not cits_ok:
                    status += f" [Wrong Citations: got {list(cits)}, expected {list(expected_cits)}]"

            total_score += score
            print(f"[{query_id}] {status} (time: {elapsed:.3f}s)")
            print(f"         answer='{ans}', citations={list(cits)}")

        print(f"\n==========================================")
        print(f"FINAL SCORE: {total_score} / {max_score}")
        print(f"==========================================")
        return 0 if total_score == max_score else 1


if __name__ == "__main__":
    sys.exit(main())
