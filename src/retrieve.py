import json, faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNKS_PATH = "data/chunks_200.json"
INDEX_PATH = "data/index_200.faiss"

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, encoding="utf-8") as f:
    chunks = json.load(f)


def retrieve(question, k=5):
    q = model.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    return [{**chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0])]


if __name__ == "__main__":
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