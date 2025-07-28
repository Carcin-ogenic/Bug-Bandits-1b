#!/usr/bin/env bash
set -euo pipefail

# absolute internal paths
PDF_DIR=/app/input/pdfs
OUTLINES=/app/output/outlines          # <– JSONs land here
MODEL=/app/models/classifier.pkl
CACHE=/app/models/cache.npz
CHAL=/app/input/challenge1b_input.json
OUT_JSON=/app/output/challenge1b_output.json

mkdir -p "$OUTLINES"

echo "▶ 1/3  extraction"
python /app/src/extraction.py "$PDF_DIR" "$OUTLINES" --model_pickle "$MODEL"

echo "▶ 2/3  cache building"
python /app/src/build_cache.py "$OUTLINES" "$CACHE"

echo "▶ 3/3  ranking"
python /app/src/rank.py "$CHAL" "$CACHE" "$OUT_JSON"

echo "✓ pipeline finished — see $OUT_JSON"
