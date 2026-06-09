"""Query-time retrieval over the chunk-level ChromaDB index (Ship D).

Thin wrapper that turns a user query string into a clean list of chunk hits:
embed the query -> VectorStore.search_similar -> flatten ChromaDB's nested
result. Everything downstream (qa.py, app.py) consumes the flat hit dicts and
never touches ChromaDB's `[0]` nesting.
"""

from typing import List

from src.processing.embeddings import EmbeddingGenerator
from src.storage.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: EmbeddingGenerator, vstore: VectorStore) -> None:
        # Reuse the SAME instances the pipeline already built — don't reload the
        # embedding model. (In app.py / qa.py these come from the shared singletons.)
        self.embedder = embedder
        self.vstore = vstore

    def retrieve(self, query: str, top_k: int) -> List[dict]:
        query_embedding = self.embedder.generate_embedding(query).tolist()
        result = self.vstore.search_similar(query_embedding, n_results=top_k)

        # ChromaDB nests each field one level deep (one inner list per query; we
        # sent a single query). Flatten the [0] here so nothing downstream sees it.
        ids = result["ids"][0]
        documents = result["documents"][0] # type: ignore
        metadatas = result["metadatas"][0] # type: ignore
        distances = result["distances"][0] # type: ignore

        hits = []
        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            hits.append({
                "chunk_id": chunk_id,
                "article_id": meta["article_id"],
                "text": text,
                "title": meta["title"],
                "source": meta["source"],
                "url": meta["url"],
                "published_at": meta["published_at"],
                "distance": distance,
            })
        return hits


if __name__ == "__main__":
      from src.config import Settings
      settings = Settings()  # type: ignore
      r = Retriever(EmbeddingGenerator(settings), VectorStore(settings))
      for h in r.retrieve("federal reserve interest rates", settings.RETRIEVAL_TOP_K):
          print(f"{h['distance']:.4f}  {h['title']}  [{h['chunk_id']}]")