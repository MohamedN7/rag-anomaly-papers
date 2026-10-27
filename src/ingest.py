import arxiv, json, os, time
import urllib.request

os.makedirs("data/pdfs", exist_ok=True)

queries = [
    "time series anomaly detection",
    "multivariate time series anomaly detection",
    "autoencoder anomaly detection time series",
    "transformer anomaly detection time series",
    "anomaly detection industrial sensor time series",
    "time series anomaly detection benchmark evaluation",
    "unsupervised anomaly detection LSTM time series",
]
MAX_PAPERS = 45


def is_relevant(r):
    """Garde l'article seulement s'il parle de séries temporelles ET d'anomalies."""
    text = (r.title + " " + r.summary).lower()
    has_ts = any(k in text for k in ["time series", "time-series", "timeseries"])
    has_anomaly = "anomal" in text or "outlier" in text
    return has_ts and has_anomaly


client = arxiv.Client()
metadata = {}

for q in queries:
    if len(metadata) >= MAX_PAPERS:
        break
    search = arxiv.Search(query=q, max_results=30,
                          sort_by=arxiv.SortCriterion.Relevance)
    for r in client.results(search):
        if len(metadata) >= MAX_PAPERS:
            break
        pid = r.get_short_id().replace("/", "_")
        if pid in metadata:
            continue
        if not is_relevant(r):
            print("SKIP", pid, r.title)
            continue

        path = f"data/pdfs/{pid}.pdf"
        try:
            if not os.path.exists(path):
                urllib.request.urlretrieve(r.pdf_url, path)
                time.sleep(3)
        except Exception as e:
            print("ERREUR", pid, e)
            continue

        metadata[pid] = {
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "year": r.published.year,
            "url": r.entry_id,
        }
        print("OK", pid, r.title)

with open("data/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n{len(metadata)} articles retenus")