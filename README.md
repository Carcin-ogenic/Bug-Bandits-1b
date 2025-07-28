# Bug-Bandits-1b

A CPU-only, offline-capable pipeline for hierarchy-aware PDF outline extraction and reranking.

## Prerequisites

- Docker installed on an AMD64 (x86_64) machine
- No GPU or internet required during runtime

## Repository Layout

At the root of the repository, you should have:

```
.
├── input/
│   ├── pdfs/                    # place all your PDFs here
│   └── challenge1b_input.json   # copy the challenge spec here
└── output/
    └── outlines/                # this will be populated by the container
```

## Build the Docker Image

```bash
docker build --platform linux/amd64 -t bug-bandits-1b .
```

## Run the Complete Offline Pipeline

```bash
docker run --rm \
  --mount type=bind,source="$(pwd)/input",target=/app/input,readonly \
  --mount type=bind,source="$(pwd)/output",target=/app/output \
  --network none \
  bug-bandits-1b
```

## Results

After the container finishes, you will find:

- `output/outlines/*.json`  
  &nbsp;&nbsp;Generated outline for each PDF
- `output/challenge1b_output.json`  
  &nbsp;&nbsp;Final ranked sections & subsections

## Performance

- Completes in under 60 seconds on a typical modern CPU
- Fully offline—no external network calls during extraction or ranking
