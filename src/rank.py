#!/usr/bin/env python3
"""
rank_pages.py  –  produces challenge1b_output.json
usage:
    python rank_pages.py input.json heading_cache.npz output.json
"""
import json, sys, time, numpy as np
from collections import defaultdict
from utils import norm, embed_dense, cross_score
import sys
from pathlib import Path  # already imported

# ensure root folder is on path so we can import headings.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from headings import extract_pages_with_headings

WEIGHT = {"TITLE": 3, "H1": 2, "H2": 1.5, "H3": 1.0}
TOP_CANDIDATES = 120
TOP_SECTIONS = 5
TOP_SUBSECS = 5


def load_cache(cache_npz: Path):
    z = np.load(cache_npz, allow_pickle=True)
    # retrieve document-level titles mapping
    titles = None
    if "titles" in z:
        # stored as 0-d object array containing dict
        titles = z["titles"].item()
    return z["emb"], z["meta"], titles


def cosine(mat, vec):           # supports mat:(N, D) or (D,), vec:(D,)
    vec_norm = np.linalg.norm(vec)
    mat = np.asarray(mat)
    if mat.ndim == 1:
        return np.dot(mat, vec) / (np.linalg.norm(mat) * vec_norm)
    # mat is 2D
    norms = np.linalg.norm(mat, axis=1)
    return (mat @ vec) / (norms * vec_norm)


def main(chal_json, cache_npz, out_json):
    t0 = time.time()
    emb, meta, titles = load_cache(Path(cache_npz))   # meta entries: (pdf, page, lvl, text), titles: dict
    chal = json.load(Path(chal_json).open(encoding="utf-8"))

    # prepare query vector
    query_txt = norm(chal["persona"]["role"] + " " + chal["job_to_be_done"]["task"])
    qv = embed_dense([query_txt])[0]

    # -------- dense retrieval ------------------------------------------
    cos_scores = cosine(emb, qv)                       # (N,)
    top_idx = np.argpartition(cos_scores, -TOP_CANDIDATES)[-TOP_CANDIDATES:]

    # -------- re-rank with cross-encoder -------------------------------
    pairs = []
    for i in top_idx:
        pdf, page, lvl, text = meta[i]
        context = f"{pdf} {lvl} {text}"
        pairs.append((query_txt, context))
    ce_scores = cross_score(pairs)
    # if cross_score returned full-length array, slice to top candidates
    if ce_scores.shape[0] != top_idx.shape[0]:
        ce_scores = ce_scores[top_idx]

    # sort by descending cross-encoder score
    order = np.argsort(-ce_scores)
    final_idx = top_idx[order]

    # -------- aggregate to sections & pages ----------------------------
    # map each top candidate to its cross-encoder score
    ce_score_map = {i: score for i, score in zip(top_idx, ce_scores)}
    # collect weighted scores per (pdf, page)
    section_scores = defaultdict(list)
    for i in final_idx:
        pdf, page, lvl, _ = meta[i]
        weight = WEIGHT.get(lvl, 1.0)
        score = ce_score_map.get(i, cosine(emb[i], qv))
        section_scores[(pdf, page)].append(score * weight)
    # pick sections by highest individual weighted score
    best_secs = sorted(section_scores.items(), key=lambda kv: -max(kv[1]))[:TOP_SECTIONS]

    # gather subsection candidates (children on same pdf >= page)
    sub_cand = []
    for (pdf, page), _score in best_secs:
        for i, (p, pn, lvl, text) in enumerate(meta):
            if p == pdf and pn >= page and lvl in {"H2", "H3"}:
                if i in top_idx:
                    score = ce_scores[list(top_idx).index(i)]
                else:
                    score = cosine(emb[i], qv)
                sub_cand.append((i, score))

    # sort all subsection candidates by score
    sub_sorted = sorted(sub_cand, key=lambda t: -t[1])

    # -------- build output JSON ----------------------------------------
    out = {
        "metadata": {
            "input_documents": [d["filename"] for d in chal["documents"]],
            "persona": chal["persona"]["role"],
            "job_to_be_done": chal["job_to_be_done"]["task"],
            "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    # fill extracted sections: choose highest-scoring H1 per section
    for rank, ((pdf, page), _) in enumerate(best_secs, start=1):
        title = ""
        # find the top H1 meta entry for this (pdf, page)
        for i in final_idx:
            m_pdf, m_page, m_lvl, m_text = meta[i]
            if m_pdf == pdf and m_page == page and m_lvl == "H1":
                title = m_text
                break
        # fallback to document title
        if not title and titles:
            title = titles.get(pdf, "")
        out["extracted_sections"].append({
            "document": f"{pdf}.pdf",
            "section_title": title,
            "importance_rank": rank,
            "page_number": int(page)
        })

    # fill subsection analysis
    pdf_dir = Path(chal_json).parent / "pdfs"
    seen = set()
    for idx, _ in sub_sorted:
        pdf, page, lvl, heading = meta[idx]
        # dedupe by unique heading within a document
        key = (pdf, heading)
        if key in seen:
            continue
        seen.add(key)
        pdf_path = pdf_dir / f"{pdf}.pdf"
        pages = extract_pages_with_headings(pdf_path)
        # get the chunk for this subsection
        refined = ""
        for p in pages:
            if p["page_number"] == page and p["section_title"] == heading:
                refined = p["refined_text"]
                break
        # fallback to first chunk on page
        if not refined and 1 <= page <= len(pages):
            refined = pages[page-1]["refined_text"]
        out["subsection_analysis"].append({
            "document": f"{pdf}.pdf",
            "refined_text": refined,
            "page_number": int(page)
        })
        if len(out["subsection_analysis"]) >= TOP_SUBSECS:
            break

    # write output
    with Path(out_json).open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("✅ wrote", out_json, f"in {time.time() - t0:4.1f}s")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: rank_pages.py <challenge.json> <cache.npz> <output.json>")
    main(*sys.argv[1:])
