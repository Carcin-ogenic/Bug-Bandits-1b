#!/usr/bin/env python3
import os
import json
import pickle
import argparse
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import re
import statistics
from collections import defaultdict
from pathlib import Path
import fitz  # PyMuPDF

# Regex for numbering detection
NUM_RE = re.compile(r"^\s*(\d+(\.\d+)*|[IVXLC]+\.)\s")

# ——— 1) Heading grouping logic ———
def compute_gap_stats(lines):
    y_gaps, prev_page, prev_y0 = [], None, None
    for ln in sorted(lines, key=lambda x: (x["page"], -x["y0"])):
        if prev_page == ln["page"] and prev_y0 is not None:
            y_gaps.append(abs(prev_y0 - ln["y0"]))
        prev_page, prev_y0 = ln["page"], ln["y0"]
    return statistics.median(y_gaps) if y_gaps else 14

def heading_similarity(a, b, gap_med):
    # 🚫 Hard guards to avoid paragraph merging
    if b["indent"] - a["indent"] > 30:
        return 0
    if re.match(r"^[a-z]", b["text"]) or re.search(r"[.:]$", b["text"]):
        return 0

    # 🚫 New: Prevent merging across separate columns at similar y-levels
    if abs(a["y0"] - b["y0"]) < 20 and abs(a["x0"] - b["x0"]) > 100:
        return 0
    
    # 🚫 Prevent merging if line1 is very long (likely a paragraph)
    if a["line_len"] > 100:
        return 0
    
    # 🚫 Prevent merging if line1 ends with sentence-ending punctuation
    if re.search(r'[.!?]$', a["text"].strip()):
        return 0
    
    # 🚫 Prevent merging if there's a significant size difference (heading vs body text)
    size_diff = abs(a["size"] - b["size"])
    if size_diff > 2.5:
        return 0
    
    # 🚫 Prevent merging if line2 starts with a number/bullet (likely new item)
    if re.match(r'^\s*(\d+\.|\d+\)|\•|\-)', b["text"]):
        return 0

    vert_gap = abs(a["y0"] - b["y0"])
    indent_sim = abs(a["indent"] - b["indent"])
    size_sim = abs(a["size"] - b["size"])
    font_sim = (a["font_hash"] == b["font_hash"])
    short_lines = max(a["line_len"], b["line_len"]) < 60
    caps_sim = min(a["caps"], b["caps"]) > 0.15
    
    # 🎯 Stricter vertical gap requirement
    vert_gap_ok = 0 < vert_gap < gap_med * 0.8  # Tightened from 1.25 to 0.8

    score = 0
    if vert_gap_ok:
        score += 2  # More weight on vertical proximity
    if indent_sim < 50:  # Tightened from 150 to 50
        score += 1
    if size_sim < 1.2:  # Tightened from 1.7 to 1.2
        score += 1
    if font_sim:
        score += 1
    if short_lines:
        score += 1
    if caps_sim:
        score += 1
    return score

def fuzzy_group_headings(lines):
    gap_med = compute_gap_stats(lines)
    lines = sorted(lines, key=lambda x: (x["page"], -x["y0"], x["x0"]))
    used, groups, i, n = set(), [], 0, len(lines)
    while i < n:
        if i in used:
            i += 1
            continue
        run, curr, j = [lines[i]], lines[i], i+1
        while j < n and lines[j]["page"] == curr["page"]:
            if heading_similarity(curr, lines[j], gap_med) >= 5:  # Changed from 4 to 5 for stricter merging
                run.append(lines[j])
                used.add(j)
                curr = lines[j]
                j += 1
            else:
                break
        if len(run) > 1:
            # Improved text merging for multi-line headings
            texts = []
            for r in run:
                clean_text = r["text"].strip()
                if clean_text:
                    texts.append(clean_text)
            
            merged_text = " ".join(texts)
            # Final normalization to ensure consistent spacing
            merged_text = re.sub(r'\s+', ' ', merged_text).strip()
            
            # 🎯 Additional check: if merged text is too long, don't merge
            if len(merged_text) > 150:  # Prevent very long merged lines
                groups.extend(run)  # Add individual lines instead of merging
            else:
                groups.append({
                    **run[0],
                    "text": merged_text,
                    "y0": min(r["y0"] for r in run),
                    "line_len": len(merged_text),
                    "caps": sum(c.isupper() for c in merged_text) / (len(merged_text) or 1)
                })
        else:
            groups.append(run[0])
        i += len(run)
    return groups

# ——— 2) Extract & group lines via PyMuPDF (same logic as training) ———
def iter_lines(pdf_path):
    all_lines = []
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    
    for page_i in range(len(doc)):
        page = doc[page_i]
        line_objs = []
        
        # Get text blocks with detailed formatting information
        blocks = page.get_text("dict")
        
        for block in blocks["blocks"]:
            if "lines" not in block:  # Skip image blocks
                continue
                
            for line in block["lines"]:
                if not line["spans"]:  # Skip empty lines
                    continue
                
                # Collect all characters/spans in this line
                all_chars = []
                full_text = ""
                
                for span in line["spans"]:
                    text = span["text"]
                    if not text.strip():
                        continue
                    
                    # Preserve spacing but normalize multiple spaces
                    full_text += text
                    
                    # Create character-like objects for compatibility
                    char_info = {
                        "fontname": span["font"],
                        "size": span["size"],
                        "x0": span["bbox"][0],
                        "y0": span["bbox"][1],
                        "x1": span["bbox"][2],
                        "y1": span["bbox"][3],
                        "text": text
                    }
                    all_chars.append(char_info)
                
                # Normalize text spacing but preserve structure
                full_text = re.sub(r'\s+', ' ', full_text.strip())
                if len(full_text) < 3:
                    continue
                
                if not all_chars:
                    continue
                
                # Extract font information
                fontnames = set(char["fontname"] for char in all_chars)
                
                # Calculate positions (PyMuPDF uses different coordinate system)
                # Convert to pdfminer-like coordinates for consistency
                page_height = page.rect.height
                y0 = page_height - max(char["y1"] for char in all_chars)  # Flip Y coordinate
                x0 = min(char["x0"] for char in all_chars)
                x1 = max(char["x1"] for char in all_chars)
                
                # Detect bold fonts (similar logic to pdfminer version)
                bold = int(any("Bold" in char["fontname"] or "Black" in char["fontname"] or 
                             "Heavy" in char["fontname"] or "Demi" in char["fontname"] 
                             for char in all_chars))
                
                line_objs.append({
                    "text": full_text,
                    "size": max(char["size"] for char in all_chars),
                    "bold": bold,
                    "indent": x0,
                    "caps": sum(map(str.isupper, full_text)) / len(full_text),
                    "num": int(bool(NUM_RE.match(full_text))),
                    "line_len": len(full_text),
                    "page": page_i,
                    "y0": y0,
                    "x0": x0,
                    "x1": x1,
                    "font_hash": hash(tuple(sorted(fontnames))) & 0xFFFFFFFF,
                })
        
        # Horizontal merge (same logic as original)
        clusters_by_y0 = defaultdict(list)
        for line in line_objs:
            key = round(line["y0"] / 2)
            clusters_by_y0[key].append(line)
        merged_lr_lines = []
        for group in clusters_by_y0.values():
            group = sorted(group, key=lambda x: x["x0"])
            # Improved text merging with proper spacing
            texts = []
            for part in group:
                clean_text = part["text"].strip()
                if clean_text:
                    texts.append(clean_text)
            
            full_text = " ".join(texts)
            # Final normalization to ensure consistent spacing
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            base = group[0]
            base.update({
                "text": full_text,
                "line_len": len(full_text),
                "caps": sum(map(str.isupper, full_text)) / len(full_text) if len(full_text) else 0,
                "x1": group[-1]["x1"]
            })
            merged_lr_lines.append(base)
        all_lines.extend(merged_lr_lines)
    
    doc.close()
    
    # Robust vertical merge (same logic as original)
    merged_lines = fuzzy_group_headings(all_lines)
    for ln in merged_lines:
        yield ln

# ——— 3) Embedding helper ———
def load_text_embedder(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    session = ort.InferenceSession(
        f"{model_dir}/model_quantized.onnx",
        providers=["CPUExecutionProvider"]
    )
    input_names = {inp.name for inp in session.get_inputs()}

    def embed(text: str) -> np.ndarray:
        toks = tokenizer(text, return_tensors="np", padding=True, truncation=True)
        ort_in = {}
        for k, v in toks.items():
            if k in input_names:
                if v.dtype == np.int32:
                    v = v.astype(np.int64)
                ort_in[k] = v
        out = session.run(None, ort_in)[0]
        if out.ndim == 3:
            out = out[0]
        return out.mean(axis=0)
    return embed

# ——— 4) Main: label PDFs and write JSON ———
def main(pdf_dir, output_dir, model_pickle):
    # load classifier bundle
    data = pickle.load(open(model_pickle, "rb"))
    clf, scaler, num_cols = data["clf"], data["scaler"], data["num_cols"]
    embed_text = load_text_embedder(data["model_dir"])

    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, fname)
        # 1) extract & group
        lines = list(iter_lines(path))
        # 2) compute per‐PDF median size & rel_size (same as training)
        sizes = [ln["size"] for ln in lines]
        m = sorted(sizes)[len(sizes)//2] if sizes else 1.0
        
        # 3) build feature arrays
        # 3a) text embeddings
        embs = np.vstack([embed_text(ln["text"]) for ln in lines])
        
        # 3b) numeric feature normalizations (must match training exactly!)
        X_num = []
        for ln in lines:
            row = [
                ln["size"] / m,  # rel_size
                ln["bold"],      # bold
                ln["indent"] / 600,  # indent (normalized same as training)
                ln["caps"],      # caps
                ln["num"],       # numbered
                ln["line_len"],  # line_len
                ln["page"],      # page
                ln["y0"] / 800,  # y0_pos (normalized same as training)
                ln["font_hash"]  # font_hash
            ]
            X_num.append(row)
        X_num = np.array(X_num)
        X_num_scaled = scaler.transform(X_num)
        X = np.hstack([embs, X_num_scaled])

        # 4) predict labels
        preds = clf.predict(X)

        # 5) assemble JSON
        titles = [ln["text"] for ln, p in zip(lines, preds) if p == "TITLE"]
        title = titles[0] if titles else ""
        outline = [
            {"level": p, "text": ln["text"], "page": ln["page"]}
            for ln, p in zip(lines, preds)
            if p in ("H1", "H2", "H3", "H4")
        ]

        # 6) write JSON
        base, _ = os.path.splitext(fname)
        with open(os.path.join(output_dir, f"{base}.json"), "w", encoding="utf-8") as fh:
            json.dump({"title": title, "outline": outline}, fh, ensure_ascii=False, indent=2)

        print(f"Generated {base}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_dir", help="Input PDFs directory")
    parser.add_argument("output_dir", help="Where to write JSON outlines")
    parser.add_argument(
        "--model_pickle",
        default="./models/classifier.pkl",
        help="Path to trained classifier bundle"
    )
    args = parser.parse_args()
    main(args.pdf_dir, args.output_dir, args.model_pickle)
