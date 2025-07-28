#!/usr/bin/env python3
"""
Challenge-1B hierarchy-aware reranker
──────────────────────────────────────
 1. Score every heading:
       ALPHA_H * cos(heading) +
       ALPHA_C * cos(chunk)   +
       BETA_CE * CE(query, context)           (for TOP_K_CE only)

 2. Keep SECTIONS_FOR_SCAN best H1/TITLE parents
 3. Rescore *all* H2/H3 under those parents (dense only)
 4. Return
       • 5 parents with highest-scoring child  → extracted_sections
       • 5 highest-scoring UNIQUE children    → subsection_analysis
"""

import json, sys, time
from pathlib import Path
from typing  import Tuple, List

import numpy as np
from utils    import norm, embed_dense, cross_score
from headings import extract_pages_with_headings

# ─── tunables (stay < 50 s) ───────────────────────────────────────────────
TOP_K_DENSE       = 140       # headings kept after cosine filter
TOP_K_CE          = 60         # pairs sent to cross encoder
SECTIONS_FOR_SCAN = 40        # parents kept for child scan
ALPHA_H , ALPHA_C = 0.15, 0.20
BETA_CE           = 0.65
LEVEL_W           = {"TITLE":1.9, "H1":1.6, "H2":1.3, "H3":1.0}
MAX_SECTIONS_OUT  = 5
# ───────────────────────────────────────────────────────────────────────────

def cosine(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return (mat @ vec) / (np.linalg.norm(mat, 1) * np.linalg.norm(vec) + 1e-8)

def load_cache(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return z["emb"].astype("float32"), z["meta"]        # meta = (pdf,page,lvl,text)

def collapse(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)
    while a.ndim > 1:                       # (B,k,dim,… ) → (B,)
        a = a.mean(axis=-1)
    return a

# ─── main ─────────────────────────────────────────────────────────────────
def main(chal_json: str, cache_npz: str, out_json: str) -> None:
    t0 = time.time()
    spec     = json.load(open(chal_json, encoding="utf-8"))
    pdf_dir  = Path(chal_json).parent / "pdfs"
    persona  = spec["persona"]["role"]
    job      = spec["job_to_be_done"]["task"]
    query    = norm(f"{persona} {job}")

    q_vec    = embed_dense([query])[0]

    emb_h, meta = load_cache(Path(cache_npz))
    k_dense = min(TOP_K_DENSE, len(emb_h))
    cos_h   = cosine(emb_h, q_vec)

    idx_dense = np.argpartition(-cos_h, k_dense-1)[:k_dense]

    # ── load & embed page chunks once ────────────────────────────────────
    pages_cache, chunks = {}, []
    for idx in idx_dense:
        pdf, page0, *_ = meta[idx]
        if pdf not in pages_cache:
            try:
                pages_cache[pdf] = extract_pages_with_headings(pdf_dir/f"{pdf}.pdf")
            except Exception:
                pages_cache[pdf] = []
        pages = pages_cache[pdf]
        chunk = pages[page0]["refined_text"] if 0 <= page0 < len(pages) else ""
        chunks.append(chunk)

    cos_c = cosine(embed_dense(chunks), q_vec)

    # ── dense score = heading + chunk ────────────────────────────────────
    h_norm = (cos_h[idx_dense]-cos_h[idx_dense].min())/(cos_h[idx_dense].ptp()+1e-8)
    c_norm = (cos_c          -cos_c.min())            /(cos_c.ptp()+1e-8)
    dense  = ALPHA_H*h_norm + ALPHA_C*c_norm

    # ── cross-encoder on best TOP_K_CE titles+chunks ─────────────────────
    # prepend full persona+job and include PDF filename for semantic filtering
    ce_query = f"{persona}: {job}"
    ce_pairs = [(
        ce_query,
        f"{meta[i][2]} {meta[i][3]}. PDF={meta[i][0]}. Context: {chunks[p][:2000]}"
    ) for p,i in enumerate(idx_dense[:TOP_K_CE])]
    ce_raw   = collapse(cross_score(ce_pairs))
    ce_norm  = (ce_raw - ce_raw.min()) / (ce_raw.ptp()+1e-8)
    # clamp cross-encoder output to TOP_K_CE length to avoid shape mismatch
    ce_norm  = np.asarray(ce_norm).flatten()[:TOP_K_CE]
    fused    = dense.copy()
    fused[:TOP_K_CE] = (1-BETA_CE)*dense[:TOP_K_CE] + BETA_CE*ce_norm

    # ── helper to find nearest parent H1/TITLE in the dense list ─────────
    idx_list = idx_dense.tolist()
    def parent_key(i:int)->Tuple[str,int,str]:
        pdf,page,lvl,text = meta[i]
        if lvl in ("TITLE","H1"): return (pdf,page,text)
        try:
            pos = idx_list.index(i)
        except ValueError:
            # index not in dense list; fallback to scan previous meta entries
            for j in range(i-1, -1, -1):
                p_pdf, p_page, p_lvl, p_text = meta[j]
                if p_pdf == pdf and p_lvl in ("TITLE","H1"):
                    return (p_pdf, p_page, p_text)
            return (pdf, page, text)
        for j in range(pos-1,-1,-1):
            p_pdf,p_page,p_lvl,p_text = meta[idx_dense[j]]
            if p_pdf==pdf and p_lvl in ("TITLE","H1"):
                return (p_pdf,p_page,p_text)
        return (pdf,page,text)

    # ── weighted heading scores -----------------------------------------
    head_score = {idx: fused[p] * LEVEL_W.get(meta[idx][2],1.0)
                  for p,idx in enumerate(idx_dense)}

    # ── choose SECTIONS_FOR_SCAN top parents ----------------------------
    parent_scores = {}
    for idx,score in head_score.items():
        key = parent_key(idx)
        if score > parent_scores.get(key, -1):
            parent_scores[key] = score
    parents = sorted(parent_scores.items(),
                     key=lambda kv:-kv[1])[:SECTIONS_FOR_SCAN]
    parent_set = {k for k,_ in parents}

    # ── collect every child under those parents -------------------------
    child_idx = [i for i in range(len(meta))
                 if parent_key(i) in parent_set and meta[i][2] in ("H2","H3")]

    # dense rescoring for *new* children (chunk already cached if same page)
    new_chunks, new_ids = [], []
    for idx in child_idx:
        pdf,page,_,_ = meta[idx]
        pages = pages_cache.get(pdf)
        if pages is None:
            try:
                pages = pages_cache[pdf] = extract_pages_with_headings(pdf_dir/f"{pdf}.pdf")
            except: pages=[]
        new_chunks.append(pages[page]["refined_text"] if 0<=page<len(pages) else "")
        new_ids.append(idx)

    cos_new_c = cosine(embed_dense(new_chunks), q_vec) if new_chunks else np.array([])
    for j,idx in enumerate(new_ids):
        h = (cos_h[idx]-cos_h.min()) / (cos_h.ptp()+1e-8)
        c = (cos_new_c[j]-cos_c.min()) / (cos_c.ptp()+1e-8)
        head_score[idx] = ALPHA_H*h + ALPHA_C*c

    # ── pick FINAL 5 parents (best child wins) --------------------------
    sec_best_child = {}
    for idx,score in head_score.items():
        key = parent_key(idx)
        if key in parent_set and score > sec_best_child.get(key,(-1,-1))[1]:
            sec_best_child[key] = (idx, score)

    final_parents = sorted(sec_best_child.items(),
                           key=lambda kv:-kv[1][1])[:MAX_SECTIONS_OUT]

    extracted = []
    for rank,(k,_) in enumerate(final_parents,1):
        pdf,page,title = k
        extracted.append({
            "document"       : f"{pdf}.pdf",
            "section_title"  : title,
            "importance_rank": rank,
            "page_number"    : page+1
        })

    # ── GLOBAL best 5 unique children -----------------------------------
    all_children = [idx for idx in head_score
                    if parent_key(idx) in parent_set and meta[idx][2] in ("H2","H3")]
    all_children.sort(key=lambda i: head_score[i], reverse=True)

    used_pages:set[Tuple[str,int]] = set()
    subsections:List[dict] = []
    # pick top unique children
    for idx in all_children:
        pdf,page,_,_ = meta[idx]
        key = (pdf,page)
        if key in used_pages:
            continue
        used_pages.add(key)
        chunk = pages_cache[pdf][page]["refined_text"] if pdf in pages_cache \
                and 0<=page<len(pages_cache[pdf]) else ""
        subsections.append({
            "document"    : f"{pdf}.pdf",
            "refined_text": chunk,
            "page_number" : page+1
        })
        if len(subsections)==MAX_SECTIONS_OUT:
            break
    # fallback: if fewer than MAX_SECTIONS_OUT, fill with parent chunks
    if len(subsections) < MAX_SECTIONS_OUT:
        for sec in extracted:
            pdf = Path(sec["document"]).stem
            pg0 = sec["page_number"] - 1
            key = (pdf, pg0)
            if key in used_pages:
                continue
            pages = pages_cache.get(pdf, [])
            chunk = pages[pg0]["refined_text"] if 0 <= pg0 < len(pages) else ""
            subsections.append({
                "document": sec["document"],
                "refined_text": chunk,
                "page_number": pg0 + 1
            })
            used_pages.add(key)
            if len(subsections) == MAX_SECTIONS_OUT:
                break

    # ── JSON write -------------------------------------------------------
    out = {
        "metadata":{
            "input_documents":[d["filename"] for d in spec["documents"]],
            "persona":persona,
            "job_to_be_done":job,
            "processing_timestamp":time.strftime("%Y-%m-%dT%H:%M:%S")
        },
        "extracted_sections" : extracted,
        "subsection_analysis": subsections
    }
    json.dump(out, open(out_json,"w",encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"✓ {out_json}  {time.time()-t0:4.1f}s "
          f"({len(extracted)} sections, {len(subsections)} subsections)")

# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv)!=4:
        sys.exit("usage: python src/rank.py <challenge.json> <cache.npz> <out.json>")
    main(*sys.argv[1:])
