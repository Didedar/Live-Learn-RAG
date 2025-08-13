import httpx
from typing import List
from .config import settings
from .utils import normalize

class OllamaEmbeddings:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.embedding_model
        self.client = httpx.Client(base_url=base_url or settings.ollama_base_url, timeout=60)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = []
        for t in texts:
            r = self.client.post("/api/embeddings", json={"model": self.model, "prompt": t})
            r.raise_for_status()
            v = r.json()["embedding"]
            vecs.append(normalize(v))
        return vecs

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
