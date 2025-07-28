# Offline Hierarchy-Aware Reranker

We run three fully-offline, deterministic stages so that grading sees the same results as local runs.

## Tunables (keep runtime < 50 s)

```yaml
TOP_K_DENSE: 150 # dense cosine filter on headings
TOP_K_CE: 60 # cross-encoder rerank budget
SECTIONS_FOR_SCAN: 40 # candidate TITLE/H1 parents
ALPHA_H, ALPHA_C: 0.22, 0.23 # heading vs. chunk weight
BETA_CE: 0.55 # CE vs. dense fusion weight
LEVEL_W: # level-bias (TITLE→H3)
  TITLE: 2.5
  H1: 2.0
  H2: 1.3
  H3: 1.0
MAX_SECTIONS_OUT: 5 # final parents & subsections
```

---

## 1. PDF → Outline (`extraction.py`)

- Parse each PDF with PyMuPDF → raw page text.
- A tiny logistic-regression classifier (`models/classifier.pkl`) labels each line as TITLE/H1/H2/H3/OTHER.
- We emit one JSON per PDF under `output/outlines/` containing:
  - `title`: document title
  - `outline`: list of `{ level, text, page, y0 }` entries
- This step is ≈10 s per medium PDF and writes human-readable JSONs.

---

## 2. Outline → Dense-Vector Cache (`build_cache.py`)

- Load all headings from `output/outlines/*.json`.
- Normalize text, embed with ONNX `all-mpnet-base-v2` → 768-dim float32 matrix.
- Save to `models/cache.npz` as:
  - `emb` : (N × 768) array
  - `meta` : [(pdf_stem, page, level, text)]
  - `titles`: [heading text]
- Typical cache is < 8 MB, loads in < 200 ms.

---

## 3. Ranking (`rank.py`)

Build the query from `challenge1b_input.json` fields (persona, task, constraints) and embed once.

**Step A: Dense Filter**

- Compute
  ```
  cos_h = cosine(heading_emb, query_emb)
  cos_c = cosine(page_chunk_emb, query_emb)
  dense = ALPHA_H·norm(cos_h) + ALPHA_C·norm(cos_c)
  ```
- Keep top `TOP_K_DENSE` headings.

**Step B: Cross-Encoder Boost**

- For the top `TOP_K_CE` candidates, form CE pairs:  
  `("Persona: Task", "LEVEL TITLE. PDF=xxx. Context: chunk")`
- Run `cross_score` → normalize → `ce_norm`.
- Fuse:
  ```
  fused[i] = dense[i]                    for i ≥ TOP_K_CE
  fused[i] = (1–BETA_CE)*dense[i] + BETA_CE*ce_norm[i]  for i < TOP_K_CE
  ```

**Step C: Hierarchy Logic**

- Map each heading to its nearest higher-level parent (TITLE/H1).
- Keep the top `SECTIONS_FOR_SCAN` parents by their best child score.
- Rescore all H2/H3 under those parents (dense only).
- Return:
  - **5 parents** with highest-scoring child → `extracted_sections`
  - **5 unique children** (by PDF/page) → `subsection_analysis`

---

## Performance & Footprint

- Full pipeline on 5 medium PDFs: **≈25 s** on 8 vCPU.
- No network calls; everything in Docker image (< 1 GB, 2 × ONNX models).
- Final output: single `challenge1b_output.json` (5 sections + 5 subsections).
