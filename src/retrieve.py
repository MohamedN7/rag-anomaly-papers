"""Recherche des chunks les plus proches d'une question.

La configuration (taille, chevauchement, modèle) vient des variables d'environnement
RAG_SIZE, RAG_OVERLAP et RAG_MODEL, fixées par run_eval.py. Sans elles, c'est la
configuration de référence (200 tokens, 40 de chevauchement, MiniLM).
"""
import json, faiss
from sentence_transformers import SentenceTransformer
from rag_config import MODELS, QUERY_PREFIX, config_from_env, config_name, paths

CFG = config_from_env()
CONFIG_NAME = config_name(CFG)
MODEL_NAME = MODELS[CFG["model"]]
CHUNKS_PATH, INDEX_PATH = paths(CFG)
PREFIX = QUERY_PREFIX[CFG["model"]]

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, encoding="utf-8") as f:
    chunks = json.load(f)

if index.ntotal != len(chunks):
    raise SystemExit(f"Index ({index.ntotal}) et chunks ({len(chunks)}) désalignés : "
                     f"relance index.py pour la configuration {CONFIG_NAME}.")


def retrieve(question, k=5):
    q = model.encode([PREFIX + question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    return [{**chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0])]


if __name__ == "__main__":
    print("Configuration :", CONFIG_NAME)
    questions = [
        "How does the Anomaly Transformer detect anomalies?",
        "What is the latent anomaly indicator in Monte Carlo EM?",
        "Which datasets are used to evaluate time series anomaly detection?",
    ]
    for q in questions:
        print("\n" + "#" * 70)
        print("QUESTION :", q)
        for r in retrieve(q, k=5):
            print(f"\n[{r['score']:.3f}] {r['title'][:60]} (pages {r['page_start']}-{r['page_end']})")
            print(r["text"][:250])