"""Streamlit demo for FinNews-RAG grounded Q&A (Ship D skeleton).

This is THE demo: a query box over the chunk index that returns a grounded,
source-cited answer. Run with:  streamlit run app.py

Cold start loads the embedding model (~a few seconds) the first time a query
runs — that's the model load, not a hang.

Reuse the heavy singletons; don't reconstruct EmbeddingGenerator / VectorStore /
LLMClient per query (Streamlit re-runs this whole script top-to-bottom on every
interaction). @st.cache_resource is the idiomatic way to build them once.
"""

import streamlit as st

from src.config import Settings
from src.processing.embeddings import EmbeddingGenerator
from src.storage.vector_store import VectorStore
from src.summarization.llm_client import LLMClient
from src.rag.retriever import Retriever
from src.rag.qa import QAEngine


@st.cache_resource
def get_engine() -> QAEngine:
    """Build the QA engine once and cache it across Streamlit re-runs.

    """
    settings = Settings() # type: ignore
    embedder = EmbeddingGenerator(settings)
    vstore   = VectorStore(settings)
    llm      = LLMClient(settings)
    retriever= Retriever(embedder, vstore)
    return QAEngine(retriever, llm)
   


def main() -> None:
    """Render the grounded Q&A page: query box -> grounded answer -> sources."""
    settings = Settings()  # type: ignore
    engine = get_engine()

    st.title("FinNews-RAG")
    st.caption(
        "Ask about recent financial news — answers are grounded in retrieved "
        "articles and cited by source."
    )

    query = st.text_input("Ask a question about recent financial news")
    st.button("Ask")  # affordance only; both Enter and this button rerun the script

    # Whichever triggered the rerun, only answer once there's actual text.
    if not query.strip():
        return

    with st.spinner("Retrieving + answering..."):
        result = engine.answer_query(query, settings.RETRIEVAL_TOP_K)

    st.markdown(result.answer)

    if not result.answered_from_context:
        st.caption("answered from context: no")

    if result.citations:
        st.subheader("Sources")
        for c in result.citations:
            st.markdown(f"[{c.marker}] {c.title} — {c.source} — {c.url}")


if __name__ == "__main__":
    main()
