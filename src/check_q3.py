from retrieve import retrieve

q = "Which datasets are used to evaluate time series anomaly detection?"
for n, p in enumerate(retrieve(q, 5), 1):
    found = [kw for kw in ["UCR", "UCI", "KDD99"] if kw in p["text"]]
    print(n, p["title"][:50], "->", found)