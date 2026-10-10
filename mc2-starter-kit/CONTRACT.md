# Mini-Challenge 2 — The Submission Contract

You submit a Docker image. We run it on an AMD GPU, pass it 10 test images, and read a JSON file back for each one. This file is the contract that image has to satisfy: how your script is invoked, what it must write, and what scores zero.

## The Limits

| | | |
|---|---|---|
| Image size | **60 GiB**, uncompressed | Measured uncompressed |
| VRAM | **1 – 48 GiB**, 1% tolerance on upper bound | Sampled continuously, peak counts. Under 1 GiB means CPU fallback (disqualified) |
| Startup | **10 minutes** | Container startup and model load |
| Per image | **30 seconds** | One `--input-image` invocation, from exec to JSON file |
| Whole run | **10 minutes** | All ten images |

Ten images, **20 points each, 200 maximum**.

## Invocation Contract

For each test image:
```bash
python3 /app/app.py --input-image /app/input/image_01.png
```

Your script writes its answer to `/app/output/<input_stem>_output.json`:
```bash
/app/output/image_01_output.json
    -> {"text": "7ABC123", "confidence": 0.94}
```

## Normalization & Rules

Answers are normalized before comparison:
- Converted to uppercase
- Whitespace removed
- Characters `-`, `.`, `·`, `_` removed

### Rules:
1. **US Licence plates**: Return the plate number only — drop state names and slogans (`CALIFORNIA 7ABC123` -> `7ABC123`).
2. **Chinese Licence plates**: Keep the leading province character and letter (`京A·12345` -> `京A12345`, `沪B·88888` -> `沪B88888`).
3. **Road signs**:
   - `STOP` -> `STOP`
   - `SPEED LIMIT 65` -> `SPEED LIMIT 65`
   - `ROAD WORK AHEAD` -> `ROAD WORK AHEAD`
   - `35` -> `35`
4. Multi-line text: read top-to-bottom, joined by single spaces.
