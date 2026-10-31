"""Évaluation du RAG sur eval/questions.json.

Exemples :
    python eval/run_eval.py --no-llm                               # retrieval, configuration de référence
    python eval/run_eval.py --no-llm --size 128 --overlap 32       # retrieval, autre configuration
    python eval/run_eval.py --k-gen 8                              # génération avec 8 passages
"""
import argparse, csv, json, os, re, statistics, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rag_config import add_config_args, config_from_args, config_name

# --- options : lues AVANT d'importer retrieve, qui charge l'index de la configuration ---
parser = argparse.ArgumentParser()
add_config_args(parser)
parser.add_argument("--k-gen", type=int, default=5, help="nombre de passages donnés au LLM")
parser.add_argument("--no-llm", action="store_true", help="retrieval seulement (rapide)")
parser.add_argument("--only", default=None, help="ids séparés par des virgules, ex: q01,q36")
parser.add_argument("--tag", default=None, help="nom des fichiers de résultats")
args = parser.parse_args()

cfg = config_from_args(args)
os.environ["RAG_SIZE"] = str(cfg["size"])
os.environ["RAG_OVERLAP"] = str(cfg["overlap"])
os.environ["RAG_MODEL"] = cfg["model"]

from retrieve import retrieve, CHUNKS_PATH, MODEL_NAME, CONFIG_NAME
from generate import build_prompt, call_llm, REFUSAL, PROMPT_VERSION, LLM_MODEL

# Refus reformulés par le modèle (ex. q24 : "The paper does not specify...")
REFUSAL_HINTS = ["cannot find the answer", "does not specify", "not specified",
                 "not mentioned", "not provided", "no information"]


def is_refusal(answer):
    """Refus = phrase de refus exacte, OU réponse courte contenant une formule de refus."""
    a = " ".join(answer.lower().split())
    if REFUSAL.lower() in a:
        return True
    return len(a) < 250 and any(h in a for h in REFUSAL_HINTS)


K_LIST = [1, 3, 5, 8, 10]
K_GEN = args.k_gen
tag = args.tag or (f"{CONFIG_NAME}_retrieval" if args.no_llm
                   else f"{CONFIG_NAME}_k{K_GEN}_prompt{PROMPT_VERSION}")
OUT_JSON = os.path.join(ROOT, "eval", f"results_{tag}.json")
OUT_CSV = os.path.join(ROOT, "eval", f"results_{tag}.csv")

questions = json.load(open(os.path.join(ROOT, "eval", "questions.json"), encoding="utf-8"))
if args.only:
    wanted = set(args.only.split(","))
    questions = [q for q in questions if q["id"] in wanted]


def norm(s):
    """Minuscules et espaces normalisés, pour comparer des textes."""
    return " ".join(s.lower().split())


def is_hit_page(chunk, q):
    """Ancien critère : même article ET pages qui recouvrent une page de référence."""
    if not q["doc_id"] or chunk["doc_id"] != q["doc_id"]:
        return False
    return any(chunk["page_start"] <= p <= chunk["page_end"] for p in q["pages"])


def is_hit(chunk, q):
    """Critère actuel : même article ET le chunk contient un mot-clé de la réponse."""
    if not q["doc_id"] or chunk["doc_id"] != q["doc_id"]:
        return False
    text = norm(chunk["text"])
    return any(norm(kw) in text for kw in q["keywords"])


def retrieval_metrics(recs, key="rank"):
    recs = [r for r in recs if r["doc_id"]]
    if not recs:
        return None
    n = len(recs)
    out = {"n": n}
    for k in K_LIST:
        out[f"hit@{k}"] = sum(1 for r in recs if r.get(key) and r[key] <= k) / n
    out["MRR"] = sum(1 / r[key] for r in recs if r.get(key)) / n
    return out


def save(records, summary):
    meta = {"tag": tag, "config": CONFIG_NAME, "chunks": CHUNKS_PATH, "embedding_model": MODEL_NAME,
            "llm": None if args.no_llm else LLM_MODEL,
            "prompt_version": None if args.no_llm else PROMPT_VERSION, "k_gen": K_GEN,
            "hit_criterion": "keywords (rank) + page (rank_page)"}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "records": records},
                  f, ensure_ascii=False, indent=2)


print("Configuration :", CONFIG_NAME, "| k_gen =", K_GEN, "| fichiers :", f"results_{tag}.*")

# --- reprise : on ne refait pas les questions déjà traitées (le LLM est lent) ---
done = {}
if os.path.exists(OUT_JSON) and not args.no_llm:
    old = json.load(open(OUT_JSON, encoding="utf-8"))
    done = {r["id"]: r for r in old["records"] if r.get("answer") is not None}

records = []
for q in questions:
    if q["id"] in done:
        records.append(done[q["id"]])
        print(q["id"], "déjà fait, ignoré")
        continue

    passages = retrieve(q["question"], k=max(max(K_LIST), K_GEN))

    # rang du premier bon chunk, avec les deux critères
    rank = None
    rank_page = None
    if q["doc_id"]:
        rank = next((i + 1 for i, p in enumerate(passages) if is_hit(p, q)), None)
        rank_page = next((i + 1 for i, p in enumerate(passages) if is_hit_page(p, q)), None)

    rec = {
        "id": q["id"], "type": q["type"], "question": q["question"],
        "gold_answer": q["answer"], "doc_id": q["doc_id"], "pages": q["pages"],
        "answerable": q["answerable"], "rank": rank, "rank_page": rank_page,
        "retrieved": [{
            "rank": i + 1, "doc_id": p["doc_id"], "title": p["title"][:70],
            "pages": [p["page_start"], p["page_end"]], "score": round(p["score"], 3),
            "is_hit": is_hit(p, q),
            "is_hit_page": is_hit_page(p, q),
            "text": p["text"],
        } for i, p in enumerate(passages)],
    }

    if not args.no_llm:
        t0 = time.time()
        answer = call_llm(build_prompt(q["question"], passages[:K_GEN]))
        cites = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
        rec["llm_seconds"] = round(time.time() - t0, 1)
        rec["answer"] = answer
        rec["refused"] = is_refusal(answer)
        rec["cited"] = cites
        rec["invalid_citation"] = any(n < 1 or n > K_GEN for n in cites)
        rec["cited_hit"] = (any(is_hit(passages[n - 1], q) for n in cites if 1 <= n <= K_GEN)
                            if cites and q["doc_id"] else None)

    records.append(rec)
    save(records, None)          # sauvegarde après chaque question
    print(f'{q["id"]} {q["type"]:12s} rang={rank} rang_page={rank_page} '
          f'top1={rec["retrieved"][0]["score"]:.2f}'
          + ("" if args.no_llm else f' refus={rec["refused"]} cites={rec["cited"]}'))

# --- résumé ---
summary = {"retrieval": {}, "generation": {}}
summary["retrieval"]["all"] = retrieval_metrics(records)
summary["retrieval"]["all_page_criterion"] = retrieval_metrics(records, key="rank_page")
for t in ["precise", "numeric", "multi"]:
    summary["retrieval"][t] = retrieval_metrics([r for r in records if r["type"] == t])


def top1(rs):
    return statistics.mean(r["retrieved"][0]["score"] for r in rs) if rs else None


summary["retrieval"]["top1_score_answerable"] = top1([r for r in records if r["answerable"]])
summary["retrieval"]["top1_score_unanswerable"] = top1([r for r in records if not r["answerable"]])

gen = [r for r in records if "answer" in r]
if gen:
    ok = [r for r in gen if r["answerable"]]
    un = [r for r in gen if not r["answerable"]]
    answered = [r for r in gen if not r["refused"]]
    cited = [r for r in answered if r["cited"]]
    checkable = [r for r in cited if r["cited_hit"] is not None]
    summary["generation"] = {
        "false_refusals": f'{sum(r["refused"] for r in ok)}/{len(ok)}',
        "correct_refusals": f'{sum(r["refused"] for r in un)}/{len(un)}',
        "answers_with_citation": f"{len(cited)}/{len(answered)}",
        "invalid_citation_numbers": sum(r["invalid_citation"] for r in answered),
        "citation_points_to_gold": f'{sum(r["cited_hit"] for r in checkable)}/{len(checkable)}',
        "avg_llm_seconds": round(statistics.mean(r["llm_seconds"] for r in gen), 1),
    }

save(records, summary)

print("\n===== RETRIEVAL (questions avec article de référence) =====")
print("all = critère mots-clés | all_page_criterion = ancien critère par page")
for name, m in summary["retrieval"].items():
    if isinstance(m, dict):
        print(f'{name:18s} n={m["n"]:2d} ' + " ".join(f'hit@{k}={m[f"hit@{k}"]:.0%}' for k in K_LIST)
              + f' MRR={m["MRR"]:.2f}')
print("score top-1 moyen, questions avec réponse :", summary["retrieval"]["top1_score_answerable"])
print("score top-1 moyen, questions sans réponse :", summary["retrieval"]["top1_score_unanswerable"])
if gen:
    print("\n===== GÉNÉRATION =====")
    for k, v in summary["generation"].items():
        print(f"{k:28s} {v}")

# --- CSV pour la notation à la main (séparateur ';' pour Excel en français) ---
with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["id", "type", "question", "reponse_attendue", "reponse_systeme", "refus",
                "rang_bon_chunk", "rang_page", "citations", "citation_sur_bon_chunk",
                "note(C/P/F)", "hallucination(0/1)", "citation_ok(0/1)", "commentaire"])
    for r in records:
        w.writerow([r["id"], r["type"], r["question"], r["gold_answer"], r.get("answer", ""),
                    r.get("refused", ""), r["rank"], r.get("rank_page", ""), r.get("cited", ""),
                    r.get("cited_hit", ""), "", "", "", ""])
print("\nFichiers :", OUT_JSON, "et", OUT_CSV)