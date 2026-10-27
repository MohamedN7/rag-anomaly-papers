import pymupdf, json, os, re, unicodedata
from collections import Counter

os.makedirs("data/texts", exist_ok=True)

with open("data/metadata.json") as f:
    metadata = json.load(f)


def find_references_cut(pages):
    """Repère le dernier titre 'References' ou 'Bibliography'.
    Renvoie (index_page, position_dans_la_page) ou None."""
    pattern = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography)\s*$",
                         re.IGNORECASE | re.MULTILINE)
    cut = None
    for i, text in enumerate(pages):
        for m in pattern.finditer(text):
            cut = (i, m.start())
    return cut


def clean_pages(pages):
    # 0. normaliser l'Unicode (ligatures ﬁ -> fi, etc.)
    pages = [unicodedata.normalize("NFKC", t) for t in pages]

    # 1. couper la bibliographie
    cut = find_references_cut(pages)
    if cut is not None:
        i, pos = cut
        pages = pages[:i] + [pages[i][:pos]]

    # 2. repérer les lignes répétées (en-têtes, pieds de page)
    line_counts = Counter()
    for text in pages:
        for line in set(l.strip() for l in text.split("\n")):
            if line:
                line_counts[line] += 1
    threshold = max(3, int(0.3 * len(pages)))
    repeated = {l for l, c in line_counts.items()
                if c >= threshold and len(l) < 100}

    # 3. nettoyer chaque page
    cleaned = []
    for text in pages:
        lines = [l for l in text.split("\n")
                 if l.strip() not in repeated      # en-têtes répétés
                 and not l.strip().isdigit()]      # numéros de page seuls
        t = "\n".join(lines)
        t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)     # mots coupés
        t = re.sub(r"\s*\n\s*", " ", t)            # sauts de ligne -> espace
        t = re.sub(r"\s{2,}", " ", t).strip()
        cleaned.append(t)
    return cleaned


for pid, meta in metadata.items():
    path = f"data/pdfs/{pid}.pdf"
    if not os.path.exists(path):
        print("MANQUANT", pid)
        continue

    doc = pymupdf.open(path)
    raw_pages = [page.get_text() for page in doc]
    pages = clean_pages(raw_pages)

    out = {
        "id": pid,
        "title": meta["title"],
        "year": meta["year"],
        "url": meta["url"],
        "pages": [{"page": i + 1, "text": t} for i, t in enumerate(pages) if t.strip()],
    }
    with open(f"data/texts/{pid}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    n_chars = sum(len(t) for t in pages)
    kept = [p for p in out["pages"]]
    n_chars = sum(len(p["text"]) for p in kept)
    flag = "  <-- PEU DE TEXTE ?" if n_chars < 5000 else ""
    print(f"{pid}: {len(raw_pages)} pages, {len(kept)} non vides, {n_chars} caractères{flag}")