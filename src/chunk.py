import json, glob, bisect, statistics
from transformers import AutoTokenizer

CHUNK_TOKENS = 200   # taille d'un chunk, en tokens
OVERLAP = 40         # tokens partagés entre deux chunks consécutifs
MIN_TOKENS = 30      # on ignore les chunks trop courts
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# On utilise le tokenizer du modèle d'embeddings : on compte donc
# les tokens exactement comme lui.
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def chunk_document(doc):
    # 1. Recoller les pages en un seul texte, en retenant où commence chaque page
    full_text = ""
    page_starts, page_nums = [], []
    for p in doc["pages"]:
        page_starts.append(len(full_text))
        page_nums.append(p["page"])
        full_text += p["text"] + "\n"

    # 2. Tokeniser tout le texte, en gardant la position (en caractères) de chaque token
    enc = tokenizer(full_text, add_special_tokens=False,
                    return_offsets_mapping=True, truncation=False)
    offsets = enc["offset_mapping"]
    n = len(offsets)

    # 3. Fenêtre glissante : chaque chunk avance de (CHUNK_TOKENS - OVERLAP) tokens
    step = CHUNK_TOKENS - OVERLAP
    chunks = []
    for start in range(0, n, step):
        end = min(start + CHUNK_TOKENS, n)
        if end - start < MIN_TOKENS:
            break
        c0, c1 = offsets[start][0], offsets[end - 1][1]
        text = full_text[c0:c1].strip()

        # 4. Retrouver la page du début et de la fin du chunk
        p0 = page_nums[bisect.bisect_right(page_starts, c0) - 1]
        p1 = page_nums[bisect.bisect_right(page_starts, max(c1 - 1, 0)) - 1]

        chunks.append({
            "chunk_id": f"{doc['id']}_{len(chunks)}",
            "doc_id": doc["id"],
            "title": doc["title"],
            "year": doc["year"],
            "url": doc["url"],
            "page_start": p0,
            "page_end": p1,
            "n_tokens": end - start,
            "text": text,
        })
        if end == n:
            break
    return chunks


all_chunks = []
for path in sorted(glob.glob("data/texts/*.json")):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    all_chunks.extend(chunk_document(doc))

# ===== COLLE ICI le bloc du filtre =====
def digit_ratio(text):
    return sum(c.isdigit() for c in text) / max(len(text), 1)

# 1. Mesure : combien de chunks seraient écartés selon le seuil ?
for t in [0.10, 0.15, 0.20, 0.30]:
    n_drop = sum(digit_ratio(c["text"]) > t for c in all_chunks)
    print(f"seuil {t:.2f} : {n_drop} chunks écartés sur {len(all_chunks)}")

# 2. Filtrage
MAX_DIGIT_RATIO = 0.15
dropped = [c for c in all_chunks if digit_ratio(c["text"]) > MAX_DIGIT_RATIO]
all_chunks = [c for c in all_chunks if digit_ratio(c["text"]) <= MAX_DIGIT_RATIO]
print(f"{len(dropped)} chunks écartés, {len(all_chunks)} gardés")

# 3. Contrôle visuel : 3 chunks écartés
for c in dropped[:3]:
    print("=" * 60)
    print("ÉCARTÉ", c["chunk_id"], round(digit_ratio(c["text"]), 2))
    print(c["text"][:300])
# ===== FIN du bloc =====

out_path = f"data/chunks_{CHUNK_TOKENS}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

sizes = [c["n_tokens"] for c in all_chunks]
print(f"{len(all_chunks)} chunks écrits dans {out_path}")
print(f"tokens par chunk : min {min(sizes)}, médiane {statistics.median(sizes):.0f}, max {max(sizes)}")