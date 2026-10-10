# AMD AI League — Multi-Challenge Submission

Comprehensive repository containing implementations, test suites, and Docker containers for the **AMD AI League Hackathon** (LabLab.ai & AMD).

All solutions are built natively on the mandated **AMD ROCm PyTorch** base image (`rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`), optimized for AMD CDNA (MI300X) and RDNA (Navi 31 / RX 7900) hardware.

---

## Tournament Overview & Results

| Challenge | Focus Area | Status | Verified Benchmark Score |
| :--- | :--- | :---: | :---: |
| **Mini-Challenge 1** | AMD AI Academy Learning Pathway | Complete | Documented in [`mc1-academy/`](mc1-academy/) |
| **Mini-Challenge 2** | Robust OCR & Traffic / Plate Recognition | Complete | **200 / 200 (100%)** — avg ~0.1s / image |
| **Mini-Challenge 3** | Enterprise Multi-Hop RAG over Mixed Files | Complete | **200 / 200 (100%)** — avg ~0.04s / query |

---

## Repository Structure

```text
.
├── AMD AI League — Match 1 & 2 Challenge Brief.pdf
├── AMD AI League — Match 3 Challenge Brief.pdf
├── mc1-academy/
│   └── README.md                 # 4-step guide to complete Match 1 on AMD AI Academy
├── mc2-starter-kit/
│   ├── CONTRACT.md               # Match 2 limits, inputs, outputs, and rules
│   ├── sample-labels.json        # 10 test image ground truths
│   ├── sample-images/            # 10 test images (plates, signs, blur, glare, noise)
│   ├── test_solution.py          # Local & WSL benchmark test harness
│   ├── selfcheck.py              # Pre-submission contract validation script
│   └── starter/
│       ├── Dockerfile            # ROCm base image + daemon entrypoint
│       └── app/
│           ├── app.py            # CLI entrypoint (--input-image)
│           ├── server.py         # GPU-resident OCR daemon (/tmp/ocr.sock)
│           └── requirements.txt
└── mc3-starter-kit/
    ├── CONTRACT.md               # Match 3 limits, inputs, outputs, and rules
    ├── sample-questions.json     # 10 sample questions with expected citations & refusals
    ├── test_solution.py          # Local & WSL benchmark test harness
    ├── selfcheck.py              # Official pre-submission contract validation script
    ├── setup.sh                  # Recreates mode 000 file and empty directory
    ├── mc3-corpus/               # Sample heterogeneous corpus (PDF, DOCX, XLSX, CSV, logs, images)
    └── starter/
        ├── Dockerfile            # Safe ROCm build (--no-deps) + daemon entrypoint
        └── app/
            ├── app.py            # CLI entrypoint (--index & --query)
            ├── server.py         # GPU-resident RAG daemon (/tmp/rag.sock)
            ├── parsers.py        # Defensive parsers (catches encrypted & unreadable files)
            ├── retriever.py      # BM25 + subword tokenizer + 2-hop entity linker
            └── requirements.txt
```

---

## Architectural Highlights

### 1. Zero-Reload GPU Daemon Architecture (Overcoming the Process Trap)
The official evaluation harness spawns a fresh OS process for every evaluation item. Loading large models inside each query process causes severe timeouts against the 30-second budget. 
* Both **Match 2** and **Match 3** run a lightweight background daemon (`server.py`) started at container boot.
* The daemon keeps model weights resident in GPU VRAM (satisfying the 1–48 GiB continuous VRAM requirement).
* Invocations via `app.py` communicate over a Unix domain socket (`/tmp/ocr.sock` and `/tmp/rag.sock`) with zero process restart overhead. Query execution time dropped to **0.02 – 0.10s**.

### 2. Match 2 (OCR & License Plate Recognition)
* **License Plate Normalization**: Accurately strips US jurisdiction banners and state slogans (`CALIFORNIA`, `EXCELSIOR`, `NEW YORK`) while retaining core plate numbers (`7ABC123`, `JHT 2951`, `5XYZ891`).
* **Multilingual Integrity**: Preserves Chinese provincial characters (`京A·12345`, `沪B·88888`).
* **Road Sign Recognition**: Reads clean, noisy, blurred, and multi-line signage top-to-bottom joined by single spaces (`STOP`, `SPEED LIMIT 65`, `ROAD WORK AHEAD`, `35`).

### 3. Match 3 (Defensive Enterprise RAG)
* **Defensive Parsers**: Gracefully handles corrupt files, password-protected/encrypted PDFs (`supplier_agreement_ENCRYPTED.pdf`), unknown binaries (`telemetry_capture.dat`), and unreadable permission files (`chmod 000` / dropped `DAC_OVERRIDE`).
* **BM25 with Compound Tokenization**: Preserves hyphenated technical part numbers, ticket IDs, and error codes (`ORR-FAN-2214-B`, `ORR-1847`, `E7731`, `REV-C2`).
* **2-Hop Entity Linking**: Automatically resolves multi-hop dependencies (e.g., extracting ticket `ORR-1847` from production logs to retrieve the fixed firmware version from the bug database).
* **Exact Citation Filtering**: Enforces citation necessity over relevance, preventing over-citation penalties and executing clean empty refusals (`{"answer": "", "citations": [], "confidence": 0.0}`) for unanswerable questions.

---

## Pre-Submission Verification

Validated inside **WSL Ubuntu** on the native Linux filesystem with dropped permissions:

### Mini-Challenge 2 Verification:
```text
=== Evaluating Mini-Challenge 2 (OCR) on 10 Sample Images ===

[01] 01_us_plate_california_clean.png       -> PASS (20/20) (time: 0.266s)
[02] 02_chinese_plate_beijing_clean.png     -> PASS (20/20) (time: 0.046s)
[03] 03_us_plate_new_york_angled.jpg        -> PASS (20/20) (time: 0.187s)
[04] 04_us_plate_motion_blur.png            -> PASS (20/20) (time: 0.039s)
[05] 05_chinese_plate_shanghai_glare.jpg    -> PASS (20/20) (time: 0.054s)
[06] 06_stop_sign_clean.png                 -> PASS (20/20) (time: 0.122s)
[07] 07_stop_sign_sensor_noise.tiff         -> PASS (20/20) (time: 0.108s)
[08] 08_speed_limit_sign.jpg                -> PASS (20/20) (time: 0.256s)
[09] 09_work_zone_sign_multiline.png        -> PASS (20/20) (time: 0.138s)
[10] 10_advisory_speed_plaque.tiff          -> PASS (20/20) (time: 0.224s)

==========================================
FINAL SCORE: 200 / 200
==========================================
```

### Mini-Challenge 3 Verification:
```text
=== Step 1: Running index() ===
Index completed in 1.36s

=== Step 2: Evaluating 10 Queries ===
[query_01] PASS (20/20) (time: 0.021s) -> answer='94', citations=['specs/tq40_datasheet_r2.pdf']
[query_02] PASS (20/20) (time: 0.026s) -> answer='Q3 FY27', citations=['planning/roadmap_fy27.docx']
[query_03] PASS (20/20) (time: 0.043s) -> answer='ORR-FAN-2214-B', citations=['support/rma_parts.xlsx']
[query_04] PASS (20/20) (time: 0.059s) -> answer='4.3.2', citations=['support/bug_database.csv']
[query_05] PASS (20/20) (time: 0.039s) -> answer='E7731', citations=['logs/prod_inference_2026-09-02.log']
[query_06] PASS (20/20) (time: 0.048s) -> answer='180', citations=['engineering/ingest_service.py']
[query_07] PASS (20/20) (time: 0.050s) -> answer='B14', citations=['specs/backplane_pinout.png']
[query_08] PASS (20/20) (time: 0.046s) -> answer='REV-C2', citations=['support/asset_label.jpg']
[query_09] PASS (20/20) (time: 0.048s) -> answer='4.3.2', citations=['support/bug_database.csv', 'logs/prod_inference_2026-09-02.log']
[query_10] PASS (20/20) (time: 0.038s) -> answer='', citations=[]

==========================================
FINAL SCORE: 200 / 200
==========================================
```

---

## How to Build & Test Container Images

### Mini-Challenge 2 Container:
```bash
cd mc2-starter-kit/starter
docker build -t <your-registry>/mc2-ocr:v1 .
cd ..
python3 selfcheck.py <your-registry>/mc2-ocr:v1 sample-images/01_us_plate_california_clean.png
```

### Mini-Challenge 3 Container:
```bash
cd mc3-starter-kit/starter
docker build -t <your-registry>/mc3-rag:v1 .
cd ..
python3 selfcheck.py <your-registry>/mc3-rag:v1 mc3-corpus
```
