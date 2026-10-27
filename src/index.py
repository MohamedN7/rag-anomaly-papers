import json, faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNKS_PATH = "data/chunks_200.json"
INDEX_PATH = "data/index_200.faiss"

with open(CHUNKS_PATH, encoding="utf-8") as f:
    chunks = json.load(f)
texts = [c["text"] for c in chunks]

model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True,
                          show_progress_bar=True).astype("float32")
print("forme des embeddings :", embeddings.shape)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)
print(index.ntotal, "vecteurs écrits dans", INDEX_PATH)