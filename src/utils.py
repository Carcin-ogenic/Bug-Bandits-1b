"""
utils.py – canonicalise text, dense embedder (MiniLM-L12-quantised),
cross-encoder scorer (MPNet).
"""
import re, html, unicodedata, numpy as np
from pathlib import Path
import onnxruntime as ort
from transformers import AutoTokenizer

# ---------- canonical text --------------------------------------------------
_S = re.compile(r"\s+")
def norm(txt: str) -> str:
    """NFKC, unescape, collapse whitespace."""
    # circled1 treated as normal 1
    return _S.sub(" ", unicodedata.normalize("NFKC", html.unescape(txt))).strip()

# ---------- dense retriever (60 MB ONNX) ------------------------------------
root   = Path(__file__).resolve().parent.parent
d_dir  = root / "models" / "paraphrase-multilingual-MiniLM-L12-v2-quantized"
tok_d  = AutoTokenizer.from_pretrained(d_dir, local_files_only=True)
sess_d = ort.InferenceSession(str(d_dir/"model_quantized.onnx"),
                              providers=["CPUExecutionProvider"])
names = [out.name for out in sess_d.get_outputs()]
# print("ONNX outputs:", names)

def embed_dense(texts: list[str]) -> np.ndarray:
    bat = tok_d(texts, padding=True, truncation=True, max_length=64,
                return_tensors="np")
    feeds = {
        "input_ids":      bat["input_ids"].astype("int64"),
        "attention_mask": bat["attention_mask"].astype("int64"),
    }
    # feeds = {k: v.astype("int64") for k, v in bat.items()}
    token_embs, _ = sess_d.run(None, feeds)             # (B, L, D)
    mask = feeds["attention_mask"][..., None]           # (B, L, 1)
    pooled = (token_embs * mask).sum(1) / mask.sum(1)   # (B, D)
    return pooled      # masked mean

# ---------- heavy cross-encoder (420 MB) ------------------------------------
ce_dir = root / "models" / "all-mpnet-base-v2"
tok_ce = AutoTokenizer.from_pretrained(ce_dir, local_files_only=True)
onnx_file = ce_dir / ("model.onnx")
sess_ce = ort.InferenceSession(str(onnx_file),
                               providers=["CPUExecutionProvider"])
ce_inputs = {i.name for i in sess_ce.get_inputs()}
# torch.set_grad_enabled(False)
# torch.set_num_threads(8)          # respect 8-vCPU limit

def cross_score(pairs: list[tuple[str,str]]) -> np.ndarray:
    """
    pairs = [(query, heading), …]  →  relevance scores  (float32, higher better)
    """
    batch = tok_ce([p[0] for p in pairs], [p[1] for p in pairs],
                 padding=True, truncation=True, max_length=128,
                 return_tensors="np")
    feeds = {
        "input_ids":      batch["input_ids"].astype("int64"),
        "attention_mask": batch["attention_mask"].astype("int64"),
    }
    if "token_type_ids" in ce_inputs:
        feeds["token_type_ids"] = batch["token_type_ids"].astype("int64")

    # ONNX graph returns a (B,1) logits tensor
    logits, = sess_ce.run(None, feeds)
    # if it’s truly a single‐logit per sample, just squeeze it:
    if logits.ndim == 2 and logits.shape[1] == 1:
        scores = logits.squeeze(-1)
    # if it’s binary classification (2 logits), pick the “positive” class:
    elif logits.ndim == 2 and logits.shape[1] > 1:
        scores = logits[:, 1]
    else:
        # fallback for any other shape
        scores = logits.reshape(-1)

    return scores.astype("float32")
