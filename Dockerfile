FROM python:3.12-slim

WORKDIR /app

# PyTorch en version CPU (la version GPU pèse plusieurs Go et ne sert à rien ici)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Télécharge le modèle d'embeddings pendant la construction de l'image,
# pour que le conteneur démarre sans accès à Hugging Face
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY src/ src/
COPY app.py .
COPY data/chunks_s400_o80_bge-small.json data/index_s400_o80_bge-small.faiss data/

# Ollama tourne sur la machine hôte, pas dans le conteneur
ENV OLLAMA_URL=http://host.docker.internal:11434/api/chat

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]