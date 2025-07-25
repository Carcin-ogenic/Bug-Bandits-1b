# headings_and_chunks.py
import fitz              # pip install PyMuPDF
import re, textwrap

# ---------------------------------------------------------------------
# 1.  Heading extractor – returns one heading + a body chunk per page
# ---------------------------------------------------------------------
HEADING_SIZE_RATIO = 1.12     # font must be ≥ 12 % larger than body median
BODY_CHUNK_LINES  = 15        # how much text to keep under each heading

heading_counter = re.compile(r"^\d+(\.\d+)*\s+")     # 1 / 1.1 / …

def page_heading_and_chunk(page):
    """Return (heading, chunk_text).  Heading may be '' if none found."""
    spans   = page.get_text("dict")["blocks"]
    fonts   = [ s["size"]
                for b in spans if b["type"] == 0
                for l in b["lines"]
                for s in l["spans"] ]
    body_sz = sorted(fonts)[int(len(fonts)*0.60)] if fonts else 0     # 60 % quantile

    heading  = ""
    body_buf = []

    for block in spans:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            line_text  = "".join(s["text"] for s in line["spans"]).strip()
            if not line_text:
                continue
            max_size   = max(s["size"] for s in line["spans"])
            looks_big  = max_size >= body_sz * HEADING_SIZE_RATIO
            looks_num  = bool(heading_counter.match(line_text))

            if looks_big or looks_num:
                heading = line_text
                continue                           # skip adding the heading itself
            if len(body_buf) < BODY_CHUNK_LINES:
                body_buf.append(line_text)

    raw_chunk = " ".join(body_buf).strip()
    refined   = textwrap.fill(re.sub(r"\s+", " ", raw_chunk), width=120)

    return heading, refined

# ---------------------------------------------------------------------
# 2.  Convenience wrapper for a whole file
# ---------------------------------------------------------------------
def extract_pages_with_headings(pdf_path):
    doc   = fitz.open(pdf_path)
    pages = []
    for pno, page in enumerate(doc, 1):
        title, chunk = page_heading_and_chunk(page)
        pages.append({
            "page_number"   : pno,
            "section_title" : title,
            "refined_text"  : chunk
        })
    return pages
