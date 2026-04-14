from typing import List
import chromadb
from src.config import Settings
from database import Article



class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.financial_news_collection = self.chroma_client.get_or_create_collection(name='financial_news')
        
    
    def add_articles(self, articles: List[Article], embeddings: list):
        ids, documents, metadatas= [], [], []
        for a in articles:
            ids.append(a.url)
            documents.append(a.content)
            metadatas.append({"title": a.title, "source": a.source or "", "published_at": a.published_at or ""})
        self.financial_news_collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return 

    def search_similar(self,query_embedding: list[float], n_results=10 ):
        return self.financial_news_collection.query(query_embeddings=[query_embedding],n_results=n_results)
        
        
    
    
    


if __name__ == '__main__':
    from src.processing.embeddings import EmbeddingGenerator

    settings = Settings()  # type: ignore
    vector_store = VectorStore(settings)
    embedding_gen = EmbeddingGenerator(settings)

    # Create fake Article objects to test add_articles
    fake_articles = [
        Article(title="Fed Raises Interest Rates by 0.25%", description="Federal Reserve raises rates",
                content="The Federal Reserve raised interest rates by 25 basis points on Wednesday, signaling more hikes ahead.",
                url="https://example.com/fed-rates", source="Reuters", published_at="2026-04-14", fetched_at="2026-04-14"),
        Article(title="Apple Reports Record Q2 Earnings", description="Apple beats expectations",
                content="Apple Inc reported record quarterly revenue of $95 billion, driven by strong iPhone sales.",
                url="https://example.com/apple-earnings", source="CNBC", published_at="2026-04-14", fetched_at="2026-04-14"),
        Article(title="Bitcoin Surges Past $70,000", description="Crypto rally continues",
                content="Bitcoin surged past $70,000 for the first time as institutional investors increased their holdings.",
                url="https://example.com/bitcoin-surge", source="Yahoo Finance", published_at="2026-04-14", fetched_at="2026-04-14"),
    ]

    # Generate embeddings and add articles
    texts = [f"{a.title} {a.description}" for a in fake_articles]
    embeddings = embedding_gen.generate_embeddings(texts).tolist()
    vector_store.add_articles(fake_articles, embeddings)
    print(f"Added {len(fake_articles)} articles to vector store")

    # Search for similar articles using a query
    query = "cryptocurrency price rally"
    query_embedding = embedding_gen.generate_embedding(query).tolist()
    results = vector_store.search_similar(query_embedding, n_results=2)

    print(f"\nQuery: '{query}'")
    print(f"Top {len(results['ids'][0])} results:")
    for i, (doc_id, distance, metadata) in enumerate(zip(results['ids'][0], results['distances'][0], results['metadatas'][0])):
        print(f"  {i+1}. {metadata['title']} (source: {metadata['source']}, distance: {distance:.4f})")