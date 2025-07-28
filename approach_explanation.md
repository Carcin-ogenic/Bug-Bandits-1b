Our solution runs fully offline in three deterministic stages so that the grading server sees exactly the same environment that produced our local results.

1 . PDF → outline (extraction.py)
Each PDF is parsed with PyMuPDF to obtain the raw page text. A lightweight logistic-regression classifier (models/classifier.pkl, 60 kB) predicts whether a line is a TITLE, H1, H2, H3 or “other”. We keep only the predicted headings together with a cleaned “refined_text” version of the entire page. The output is one JSON file per PDF in output/outlines/ — small, human-readable and immune to upstream rendering quirks.

2 . outline → dense-vector cache (build_cache.py)
All heading strings are normalised (lower-case, punctuation squashed) and embedded with an ONNX copy of sentence-transformers/all-mpnet-base-v2 (no PyTorch needed). The resulting matrix «emb» (N × 768 float32) and the parallel «meta» array (pdf-stem, page, level, text) are serialised to models/cache.npz. Because headings rarely exceed 5 000 per document set, the cache occupies < 8 MB and loads in < 200 ms.

3 . ranking (rank.py)
The query is built from every information-need field in challenge1b_input.json (persona, job_to_be_done, dietary or other constraints) and embedded once.
Step A: dense filter – cosine(query, heading) plus cosine(query, page-chunk) drops the search space from N to TOP_K_DENSE (=140).
Step B: cross-encoder – a distilled mini-RoBERTa re-scores the best 60 pairs and the scores are fused back into the global list, allowing constraint-consistent headings to leap-frog early dense hits.
Step C: hierarchy logic – we keep the 40 strongest TITLE/H1 parents, rescore all H2/H3 children under them, then return the top-5 parents (“extracted_sections”) and the globally best 5 unique children (“subsection_analysis”).
The whole pass on five medium PDFs takes ≈25 s on 8 vCPU and never touches the network.

Design choices

Everything after extraction is pure NumPy so the container remains < 1 GB even with two ONNX models.

All transient artefacts (outlines/\*.json, cache.npz) stay inside the image’s /app hierarchy; only the final challenge1b_output.json is exported, keeping the host mount minimal.

The scorer is domain-agnostic; the only handcrafted bias is a mild penalty for a short list of obvious meat tokens, which can be removed without breaking other domains.
