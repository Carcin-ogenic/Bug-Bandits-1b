#!/usr/bin/env python3
"""
src/rank.py
Create Challenge-1B output from a heading cache.

Usage:
    python src/rank.py <challenge1b_input.json> <heading_cache.npz> <output.json>
"""
import sys, time, json
from pathlib import Path
from typing import Tuple, List

import numpy as np

from utils    import norm, embed_dense, cross_score
from headings import extract_pages_with_headings

# ───────────── tunables ─────────────
TOP_K_DENSE     = 120      # how many headings to keep after dense filter
ALPHA           = 0.30     # weight for dense score
BETA            = 0.70     # weight for cross‐encoder score
LEVEL_WEIGHT    = { "TITLE":3.0, "H1":2.0, "H2":1.4, "H3":1.0 }
MAX_SECTIONS    = 5
MAX_SUBSECTIONS = 5

# ───────────── helpers ─────────────
def cosine(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Row‐wise cosine(mat, vec)."""
    vec_n = np.linalg.norm(vec)
    mat_n = np.linalg.norm(mat, axis=1)
    return (mat @ vec) / (mat_n * vec_n + 1e-8)

def load_cache(npz: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(npz, allow_pickle=True)
    return z["emb"].astype("float32"), z["meta"]

# ───────────── main ─────────────
def main(chal_path: str, cache_path: str, out_path: str) -> None:
    t0 = time.time()

    # 1) load challenge spec
    chal    = json.load(open(chal_path, encoding="utf-8"))
    pdf_dir = Path(chal_path).parent / "pdfs"
    persona = chal["persona"]["role"]
    job     = chal["job_to_be_done"]["task"]
    query   = norm(f"{persona} – {job}")

    # 2) dense‐encode query
    q_vec = embed_dense([query])[0]  # (384,)

    # 3) load precomputed heading embeddings
    emb, meta = load_cache(Path(cache_path))  # emb: (N,384), meta: (N,)
    N = emb.shape[0]

    # 4) dense retrieval with precomputed norms
    k = min(TOP_K_DENSE, N)
    # compute norms once
    emb_norms = np.linalg.norm(emb, axis=1)
    q_norm = np.linalg.norm(q_vec)
    # vectorized cosine similarity
    d_scores = (emb @ q_vec) / (emb_norms * q_norm + 1e-8)
    top_idx  = np.argpartition(-d_scores, k-1)[:k]  # indices of top‐k

    # 5) cache PDFs and build (query, context) pairs for cross‐encoder
    # avoid opening the same PDF multiple times by caching pages
    pages_cache = {}
    for i in top_idx:
        pdf = meta[i][0]
        if pdf not in pages_cache:
            try:
                pages_cache[pdf] = extract_pages_with_headings(pdf_dir / f"{pdf}.pdf")
            except:
                pages_cache[pdf] = []
    pairs: List[Tuple[str,str]] = []
    refined_chunks: List[str] = []
    for i in top_idx:
        pdf, page0, lvl, title = meta[i]
        pages = pages_cache.get(pdf, [])
        chunk = pages[page0]["refined_text"] if 0 <= page0 < len(pages) else ""
        context = f"{lvl} {title}. {chunk}"
        pairs.append((query, context))
        refined_chunks.append(chunk)

    # 6) cross‐encoder scoring
    ce_out = cross_score(pairs)
    # ensure 1D array of cross-scores length k
    ce_arr = np.asarray(ce_out)
    ce_scores = ce_arr.flatten()[: len(pairs)]

    # 7) min–max normalisation
    d_sel = d_scores[top_idx]
    d_min, d_max = d_sel.min(), d_sel.max()
    d_norm = (d_sel - d_min) / (d_max - d_min + 1e-8)

    c_min, c_max = ce_scores.min(), ce_scores.max()
    c_norm = (ce_scores - c_min) / (c_max - c_min + 1e-8)

    fused = ALPHA * d_norm + BETA * c_norm  # (k,)

    # 8) apply heading‐level weights & sort
    cand = []
    for rank_i, i in enumerate(top_idx):
        pdf, page0, lvl, title = meta[i]
        w = LEVEL_WEIGHT.get(lvl, 1.0)
        cand.append((fused[rank_i] * w, pdf, page0, title, refined_chunks[rank_i]))
    cand.sort(key=lambda x: -x[0])

    # 9) select top unique sections
    extracted = []
    seen = set()
    for score, pdf, page0, title, _ in cand:
        key = (pdf, title.lower())
        if key in seen:
            continue
        seen.add(key)
        extracted.append({
            "document"       : f"{pdf}.pdf",
            "section_title"  : title,
            "importance_rank": len(extracted) + 1,
            "page_number"    : page0 + 1
        })
        if len(extracted) == MAX_SECTIONS:
            break

    # 10) gather matching chunks
    subsection_analysis = []
    for sec in extracted[:MAX_SUBSECTIONS]:
        stem = Path(sec["document"]).stem
        pg   = sec["page_number"] - 1
        # reuse cached pages if available
        pages = pages_cache.get(stem, [])
        chunk = pages[pg]["refined_text"] if 0 <= pg < len(pages) else ""
        subsection_analysis.append({
            "document"    : sec["document"],
            "refined_text": chunk,
            "page_number" : pg + 1
        })

    # 11) write output
    out = {
        "metadata": {
            "input_documents"     : [d["filename"] for d in chal["documents"]],
            "persona"             : persona,
            "job_to_be_done"      : job,
            "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        },
        "extracted_sections" : extracted,
        "subsection_analysis": subsection_analysis
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"✓ wrote {out_path} in {time.time()-t0:4.1f}s "
          f"({len(extracted)} sections, {len(subsection_analysis)} chunks)")

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python rank.py <challenge.json> <cache.npz> <output.json>")
    main(*sys.argv[1:])
