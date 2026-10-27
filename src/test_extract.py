import pymupdf  # PyMuPDF

doc = pymupdf.open("data/pdfs/2110.02642v5.pdf")  # Anomaly Transformer
print("Nombre de pages :", len(doc))
for i in range(2):
    print(f"\n--- page {i+1} ---")
    print(doc[i].get_text()[:800])