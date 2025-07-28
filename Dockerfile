# ──────────────────────────────────────────────────────────────
#  Dockerfile – Bug-Bandits-1B
#  Builds a CPU-only image that runs:
#    1) src/extraction.py       → output/outlines/*.json
#    2) src/build_cache.py      → models/cache.npz
#    3) src/rank.py             → output/challenge1b_output.json
#  ENTRYPOINT = /app/docker-entrypoint.sh

#  Usage:
#    docker build -t bug-bandits:latest .   
#   docker run --rm \
#   --mount type=bind,source="$(pwd)/input",target=/app/input \
#   --mount type=bind,source="$(pwd)/output",target=/app/output \
#   --network none \
#   bug-bandits-1b:latest
# ──────────────────────────────────────────────────────────────
# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.10-slim

# ---------- basic environment --------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/app/src

WORKDIR /app

# ---------- tiny system deps (for PyMuPDF and building wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libmupdf-dev gcc g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------- python requirements ------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- project sources & models -------------------------
# .dockerignore should exclude:  input/, output/, __pycache__, .git …
COPY src/        /app/src/
COPY models/     /app/models/
COPY input/    /app/input/
COPY output/   /app/output/

# ---------- entry-point script -------------------------------
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
