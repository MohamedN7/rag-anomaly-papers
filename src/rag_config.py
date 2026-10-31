"""Configuration partagée : taille des chunks, chevauchement, modèle d'embeddings.

Chaque configuration a un nom (ex. s200_o40_minilm) qui sert à nommer ses fichiers :
data/chunks_<nom>.json et data/index_<nom>.faiss. Ainsi, plusieurs configurations
peuvent coexister sans s'écraser.
"""
import os

# nom court -> modèle Hugging Face
MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
}

# nombre maximum de tokens lus par chaque modèle (au-delà, le texte est ignoré)
MAX_SEQ = {
    "minilm": 256,
    "bge-small": 512,
}

# préfixe ajouté aux QUESTIONS (pas aux passages), recommandé par BGE pour la recherche
QUERY_PREFIX = {
    "minilm": "",
    "bge-small": "Represent this sentence for searching relevant passages: ",
}

DEFAULT = {"size": 400, "overlap": 80, "model": "bge-small"}


def add_config_args(parser):
    """Ajoute --size, --overlap et --model à un parseur argparse."""
    parser.add_argument("--size", type=int, default=DEFAULT["size"], help="taille des chunks (tokens)")
    parser.add_argument("--overlap", type=int, default=DEFAULT["overlap"], help="chevauchement (tokens)")
    parser.add_argument("--model", default=DEFAULT["model"], choices=list(MODELS),
                        help="modèle d'embeddings")


def config_from_args(args):
    return {"size": args.size, "overlap": args.overlap, "model": args.model}


def config_from_env():
    """Lu par retrieve.py : la configuration vient des variables d'environnement
    (fixées par run_eval.py), sinon des valeurs par défaut."""
    return {
        "size": int(os.environ.get("RAG_SIZE", DEFAULT["size"])),
        "overlap": int(os.environ.get("RAG_OVERLAP", DEFAULT["overlap"])),
        "model": os.environ.get("RAG_MODEL", DEFAULT["model"]),
    }


def config_name(cfg):
    return f"s{cfg['size']}_o{cfg['overlap']}_{cfg['model']}"


def paths(cfg):
    """Chemins du fichier de chunks et de l'index pour une configuration."""
    name = config_name(cfg)
    return f"data/chunks_{name}.json", f"data/index_{name}.faiss"