import numpy as np
import tiktoken

def normalize(v):
    v = np.array(v, dtype=np.float32)
    n = np.linalg.norm(v) + 1e-12
    return (v / n).tolist()

def cosine(u, v):
    u = np.array(u, dtype=np.float32)
    v = np.array(v, dtype=np.float32)
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

def token_chunks(text: str, max_tokens: int = 400, overlap: int = 40, model="gpt-4o-mini"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        part = tokens[i:i+max_tokens]
        chunks.append(enc.decode(part))
        i += max_tokens - overlap if i + max_tokens < len(tokens) else max_tokens
    return chunks
