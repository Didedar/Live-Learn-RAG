from sqlalchemy.orm import Session
from .models import Document, Chunk
from .services_embeddings import OllamaEmbeddings
from .services_generator import OllamaLLM, build_prompt
from .utils import token_chunks, cosine

emb = OllamaEmbeddings()
llm = OllamaLLM()

def ingest_text(db: Session, text: str, metadata: dict | None = None, uri: str = "inline"):
    doc = Document(uri=uri, metadata=metadata or {})
    db.add(doc); db.flush()

    parts = token_chunks(text, max_tokens=400, overlap=40)
    vectors = emb.embed_documents(parts)

    for i, (p, v) in enumerate(zip(parts, vectors), start=1):
        ch = Chunk(document_id=doc.id, ordinal=i, content=p, embedding=v)
        db.add(ch)
    db.commit()
    return doc.id, len(parts)

def retrieve(db: Session, query: str, top_k: int = 6):
    qv = emb.embed_query(query)
    rows = db.query(Chunk).all()
    scored = []
    for ch in rows:
        if not ch.embedding:
            continue
        s = cosine(qv, ch.embedding)
        scored.append((ch, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def rag_answer(db: Session, query: str, top_k: int = 6):
    hits = retrieve(db, query, top_k=top_k)
    contexts = [{
        "document_id": ch.document_id,
        "chunk_id": ch.id,
        "score": float(score),
        "text": ch.content,
        "metadata": {}
    } for ch, score in hits]
    prompt = build_prompt(query, contexts)
    answer = llm.generate(prompt)
    return answer, contexts
