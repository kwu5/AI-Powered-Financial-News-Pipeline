"""Tests for the Ship D retriever — query embed + ChromaDB flatten.

The retriever's whole job is to embed a query, call VectorStore.search_similar,
and flatten ChromaDB's one-level-nested result into clean hit dicts. Both the
embedder and the vector store are mocked so these tests are fast and exercise
only the flattening contract — never the real model or index.
"""

from unittest.mock import MagicMock

from src.rag.retriever import Retriever


def chroma_result(n=2):
    """A ChromaDB query() result shaped exactly as the real client returns it:
    every field nested one level deep (one inner list per query)."""
    return {
        "ids": [[f"{i + 1}:{i}" for i in range(n)]],
        "documents": [[f"chunk text {i}" for i in range(n)]],
        "metadatas": [[
            {
                "article_id": i + 1,
                "title": f"Title {i}",
                "source": f"Source {i}",
                "url": f"https://example.com/{i}",
                "published_at": "2026-06-01T00:00:00Z",
            }
            for i in range(n)
        ]],
        "distances": [[round(0.1 * (i + 1), 3) for i in range(n)]],
    }


def make_retriever(search_result):
    embedder = MagicMock()
    embedder.generate_embedding.return_value.tolist.return_value = [0.0, 0.1, 0.2]
    vstore = MagicMock()
    vstore.search_similar.return_value = search_result
    return Retriever(embedder, vstore), embedder, vstore


class TestRetrieve:

    def test_flatten_produces_well_formed_hits(self):
        retriever, _, _ = make_retriever(chroma_result(2))
        hits = retriever.retrieve("federal reserve", top_k=2)

        assert len(hits) == 2
        first = hits[0]
        assert set(first) == {
            "chunk_id", "article_id", "text", "title",
            "source", "url", "published_at", "distance",
        }
        assert first["chunk_id"] == "1:0"
        assert first["article_id"] == 1
        assert first["text"] == "chunk text 0"
        assert first["title"] == "Title 0"
        assert first["source"] == "Source 0"
        assert first["url"] == "https://example.com/0"
        assert first["distance"] == 0.1

    def test_no_nesting_leaks(self):
        # Every value must be a scalar, never a [0]-wrapped list.
        retriever, _, _ = make_retriever(chroma_result(3))
        for h in retriever.retrieve("q", top_k=3):
            for v in h.values():
                assert not isinstance(v, list)

    def test_empty_result_returns_empty_list(self):
        retriever, _, _ = make_retriever(chroma_result(0))
        assert retriever.retrieve("nothing indexed", top_k=5) == []

    def test_query_is_embedded_and_top_k_forwarded(self):
        retriever, embedder, vstore = make_retriever(chroma_result(1))
        retriever.retrieve("apple earnings", top_k=7)

        embedder.generate_embedding.assert_called_once_with("apple earnings")
        # the embedded (list) query and n_results=top_k reach search_similar
        vstore.search_similar.assert_called_once_with([0.0, 0.1, 0.2], n_results=7)
