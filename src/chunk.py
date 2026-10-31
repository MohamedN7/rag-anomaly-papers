"""Découpe les articles nettoyés (data/texts/*.json) en chunks.

Exemples :
    python src/chunk.py                                  # référence : 200 tokens, 40 de chevauchement, MiniLM
    python src/chunk.py --size 128 --overlap 32
    python src/chunk.py --size 400 --overlap 80 --model bge-small
"""
import argparse, json, glob, bisect, statistics
from transformers import AutoTokenizer
from rag_config import MODELS, MAX_SEQ, add_config_args, config_from_args, config_name, paths

parser = argparse.ArgumentParser()
add_config_args(parser)
args = parser.parse_args()
cfg = config_from_args(args)

CHUNK_TOKENS = cfg["size"]
OVERLAP = cfg["overlap"]
MIN_TOKENS = 30          # on ignore les chunks trop courts
MAX_DIGIT_RATIO = 0.15   # filtre des chunks surtout numériques (tableaux, figures)

if OVERLAP >= CHUNK_TOKENS:
    raise SystemExit("Le chevauchement doit être plus petit que la taille du chunk.")
if CHUNK_TOKENS + 2 > MAX_SEQ[cfg["model"]]:
    raise SystemExit(f"{CHUNK_TOKENS} tokens + 2 tokens spéciaux dépassent la limite de "
                     f"{MAX_SEQ[cfg['model']]} du modèle {cfg['model']}.")

# Tokenizer du modèle d'embeddings : on compte les tokens exactement comme lui.
tokenizer = AutoTokenizer.from_pretrained(MODELS[cfg["model"]])


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


def digit_ratio(text):
    return sum(c.isdigit() for c in text) / max(len(text), 1)


all_chunks = []
for path in sorted(glob.glob("data/texts/*.json")):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    all_chunks.extend(chunk_document(doc))

n_before = len(all_chunks)
all_chunks = [c for c in all_chunks if digit_ratio(c["text"]) <= MAX_DIGIT_RATIO]

out_path, _ = paths(cfg)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

sizes = [c["n_tokens"] for c in all_chunks]
print(f"Configuration : {config_name(cfg)}")
print(f"{n_before} chunks avant filtre, {n_before - len(all_chunks)} écartés (chiffres > {MAX_DIGIT_RATIO:.0%})")
print(f"{len(all_chunks)} chunks écrits dans {out_path}")
print(f"tokens par chunk : min {min(sizes)}, médiane {statistics.median(sizes):.0f}, max {max(sizes)}")