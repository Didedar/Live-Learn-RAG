import httpx
from .config import settings

class OllamaLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.llm_model
        self.client = httpx.Client(base_url=base_url or settings.ollama_base_url, timeout=None)

    def generate(self, prompt: str) -> str:
        r = self.client.post("/api/generate", json={"model": self.model, "prompt": prompt, "stream": False})
        r.raise_for_status()
        return r.json().get("response", "").strip()

def build_prompt(query: str, contexts: list[dict]) -> str:
    ctx = []
    for i, c in enumerate(contexts, 1):
        ctx.append(f"[{i}] doc={c['document_id']} chunk={c['chunk_id']} score={c['score']:.3f}\n{c['text']}")
    return f"""Use ONLY the context to answer; if unsure, say you don't know.
Respond in the same language as the question and cite as [number].

Context:
{'\n\n'.join(ctx)}

Question: {query}
Answer:"""
