"""Encode les chunks d'une configuration et construit son index FAISS.

Exemples :
    python src/index.py                                  # référence
    python src/index.py --size 128 --overlap 32
    python src/index.py --size 400 --overlap 80 --model bge-small
Utilise les mêmes options que chunk.py.
"""
import argparse, json, os, faiss
from sentence_transformers import SentenceTransformer
from rag_config import MODELS, add_config_args, config_from_args, config_name, paths

parser = argparse.ArgumentParser()
add_config_args(parser)
args = parser.parse_args()
cfg = config_from_args(args)

CHUNKS_PATH, INDEX_PATH = paths(cfg)
if not os.path.exists(CHUNKS_PATH):
    raise SystemExit(f"{CHUNKS_PATH} introuvable : lance d'abord chunk.py avec les mêmes options.")

with open(CHUNKS_PATH, encoding="utf-8") as f:
    chunks = json.load(f)
texts = [c["text"] for c in chunks]

model = SentenceTransformer(MODELS[cfg["model"]])
embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True,
                          show_progress_bar=True).astype("float32")
print("Configuration :", config_name(cfg))
print("forme des embeddings :", embeddings.shape)

index = faiss.IndexFlatIP(embeddings.shape[1])   # produit scalaire = cosinus (vecteurs normalisés)
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)
print(index.ntotal, "vecteurs écrits dans", INDEX_PATH)