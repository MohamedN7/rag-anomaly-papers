import requests
from retrieve import retrieve
import os

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
LLM_MODEL = "llama3.2"
REFUSAL = "I cannot find the answer in the provided documents."

PROMPT_VERSION = 2

PROMPTS = {
    1: f"""You are a research assistant answering questions about scientific papers on time series anomaly detection.

Rules:
- Answer ONLY using the numbered passages provided. Do not use your own knowledge.
- After each claim, cite the passage(s) that support it, like [1] or [2][3].
- If the passages do not contain enough information to answer, reply exactly: "{REFUSAL}"
- Do not invent numbers, names, or results that are not in the passages.
- Answer in the same language as the question. Be concise.""",

    2: f"""You are a research assistant answering questions about scientific papers on time series anomaly detection.

Rules:
- Answer ONLY using the numbered passages provided. Do not use your own knowledge.
- Every sentence of your answer MUST end with the number(s) of the passage(s) it comes from, like [1] or [2][3]. A sentence without a citation is not allowed.
- Attribute a fact only to the paper whose passage states it. Name the paper when you report a fact. Never generalize from one paper to others.
- If the question asks for a general list or comparison and the passages only cover some papers, say which papers the information comes from and state that the list is incomplete.
- If the passages do not contain enough information to answer, reply exactly: "{REFUSAL}"
- Do not invent numbers, names, or results that are not in the passages.
- Answer in the same language as the question. Be concise.""",
}
SYSTEM_PROMPT = PROMPTS[PROMPT_VERSION]

def build_prompt(question, passages):
    blocks = []
    for n, p in enumerate(passages, 1):
        header = f"[{n}] {p['title']} (pages {p['page_start']}-{p['page_end']})"
        blocks.append(f"{header}\n{p['text']}")
    context = "\n\n".join(blocks)
    return f"Passages:\n\n{context}\n\nQuestion: {question}"


def call_llm(user_prompt):
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise SystemExit("Ollama ne répond pas. Vérifie qu'il est lancé (commande : ollama list).")
    return r.json()["message"]["content"].strip()


def ask(question, k=5):
    passages = retrieve(question, k=k)
    answer = call_llm(build_prompt(question, passages))
    return {"question": question, "answer": answer, "passages": passages}


if __name__ == "__main__":
    print("PROMPT_VERSION =", PROMPT_VERSION)
    tests = [
        "What is the latent anomaly indicator in Monte Carlo EM?",
        "How does the Anomaly Transformer detect anomalies?",
        "Which datasets are used to evaluate time series anomaly detection?",
        "What is the best recipe for chocolate cake?",
    ]
    for q in tests:
        res = ask(q)
        print("\n" + "#" * 70)
        print("QUESTION :", q)
        print("\nRÉPONSE :\n" + res["answer"])
        print("\nSOURCES RÉCUPÉRÉES :")
        for n, p in enumerate(res["passages"], 1):
            print(f"  [{n}] {p['title'][:60]} (pages {p['page_start']}-{p['page_end']}, score {p['score']:.2f})")