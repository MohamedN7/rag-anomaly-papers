import json, random

def digit_ratio(text):
    return sum(c.isdigit() for c in text) / max(len(text), 1)

chunks = json.load(open("data/chunks_200.json", encoding="utf-8"))
borderline = [c for c in chunks if 0.12 < digit_ratio(c["text"]) <= 0.20]
print(len(borderline), "chunks entre 0.12 et 0.20")

random.seed(1)
for c in random.sample(borderline, min(26, len(borderline))):
    print("=" * 60)
    print(c["chunk_id"], round(digit_ratio(c["text"]), 2))
    print(c["text"][:300])