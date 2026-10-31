"""Interface web de l'assistant RAG (Streamlit).

Lancement, depuis la racine du projet :
    streamlit run app.py
"""
import os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Configuration retenue après l'évaluation (doit être fixée AVANT d'importer retrieve)
os.environ.setdefault("RAG_SIZE", "400")
os.environ.setdefault("RAG_OVERLAP", "80")
os.environ.setdefault("RAG_MODEL", "bge-small")

import requests
import streamlit as st

st.set_page_config(page_title="Questions sur la détection d'anomalies", layout="wide")


@st.cache_resource(show_spinner="Chargement du modèle d'embeddings et de l'index...")
def load_backend():
    """Chargé une seule fois, puis gardé en mémoire entre les questions."""
    import retrieve
    import generate
    return retrieve, generate


retrieve_mod, generate_mod = load_backend()

REFUSAL_HINTS = [generate_mod.REFUSAL.lower(), "cannot find the answer", "cannot find any information",
                 "does not specify", "not specified", "not mentioned", "not provided", "no information"]

EXAMPLES = [
    "How does the Anomaly Transformer detect anomalies?",
    "How is the latent anomaly indicator modeled in the Monte Carlo EM paper?",
    "On how many real-world datasets is TAnoGAN evaluated?",
]


def is_refusal(answer):
    a = " ".join(answer.lower().split())
    if generate_mod.REFUSAL.lower() in a:
        return True
    return len(a) < 250 and any(h in a for h in REFUSAL_HINTS)


def set_question(q):
    st.session_state["question"] = q


# ---------------- Barre latérale ----------------
with st.sidebar:
    st.header("Réglages")
    k = st.slider("Nombre de passages lus par le modèle", min_value=3, max_value=8, value=5)
    show_all = st.checkbox("Ouvrir tous les passages", value=False)
    st.divider()
    st.caption("Corpus : 42 articles arXiv")
    st.caption(f"Embeddings : {retrieve_mod.MODEL_NAME}")
    st.caption(f"Configuration : {retrieve_mod.CONFIG_NAME}")
    st.caption(f"LLM : {generate_mod.LLM_MODEL} (local, via Ollama)")

# ---------------- Zone principale ----------------
st.title("Questions sur la détection d'anomalies dans les séries temporelles")
st.write("Les réponses sont construites uniquement à partir de 42 articles arXiv, avec leurs sources. "
         "Posez vos questions en anglais, la langue des articles.")

st.write("Exemples :")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(ex, on_click=set_question, args=(ex,), use_container_width=True)

question = st.text_area("Votre question", key="question", height=90)

if st.button("Poser la question", type="primary"):
    if not question.strip():
        st.warning("Écrivez une question avant de lancer la recherche.")
    else:
        with st.spinner("Recherche des passages..."):
            passages = retrieve_mod.retrieve(question, k=k)
        try:
            with st.spinner("Rédaction de la réponse (environ 30 secondes sur CPU)..."):
                answer = generate_mod.call_llm(generate_mod.build_prompt(question, passages))
            st.session_state["result"] = {"question": question, "answer": answer, "passages": passages}
        except (SystemExit, requests.exceptions.RequestException):
            st.error(f"Ollama ne répond pas à l'adresse {generate_mod.OLLAMA_URL}. "
                     "Lancez l'application Ollama, vérifiez que le modèle "
                     f"{generate_mod.LLM_MODEL} est installé, puis reposez la question.")

# ---------------- Résultat (gardé en mémoire entre deux interactions) ----------------
res = st.session_state.get("result")
if res:
    st.divider()
    st.subheader("Réponse")
    st.caption(f"Question : {res['question']}")

    refused = is_refusal(res["answer"])
    if refused:
        st.info("Les articles du corpus ne contiennent pas de quoi répondre à cette question.")
    st.markdown(res["answer"])

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", res["answer"])}
    if not refused and not cited:
        st.caption("Le modèle n'a cité aucun passage : vérifiez la réponse à l'aide des sources ci-dessous.")

    st.subheader("Sources")
    for n, p in enumerate(res["passages"], 1):
        label = f"[{n}] {p['title']} (pages {p['page_start']}-{p['page_end']})"
        if n in cited:
            label += " : citée dans la réponse"
        with st.expander(label, expanded=show_all or n in cited):
            st.markdown(f"[Ouvrir l'article sur arXiv]({p['url']})")
            st.caption(f"Similarité avec la question : {p['score']:.2f}")
            st.write(p["text"])