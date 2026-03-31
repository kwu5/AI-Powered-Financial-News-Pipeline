
from typing import Dict, List
import numpy as np
from src.config import Settings
from src.processing.embeddings import EmbeddingGenerator

class Deduplicator:
    def __init__(self, embedding_generator: EmbeddingGenerator) -> None:
        self.embedding_generator = embedding_generator
        self.settings = Settings()      # type: ignore 
        
    def cosine_similarity(self, a:np.ndarray, b:np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        
    def deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        kept = []
        kept_embeddings = []
        
        for article in articles:
            text = article["title"] + " " + article.get("description","")
            embedding = self.embedding_generator.generate_embedding(text)
            
            is_duplicate = False
            for kept_emb in kept_embeddings:
                if self.cosine_similarity(embedding, kept_emb) >=self.settings.SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(article)
                kept_embeddings.append(embedding)

        print(f"Deduplication: {len(articles)} → {len(kept)} articles ({len(articles) - len(kept)}removed)")
        return kept
        
    
if __name__ == '__main__':
    embeddingGenerator = EmbeddingGenerator(Settings())     # type: ignore
    deduplicator = Deduplicator(embeddingGenerator)
    # Identical direction → similarity ≈ 1.0
    a = np.array([0.6, 0.8, 0.0])
    b = np.array([0.6, 0.8, 0.0])
    cs1 = deduplicator.cosine_similarity(a, b)
    print(cs1)

    # Similar → similarity ≈ 0.9+
    a = np.array([0.6, 0.8, 0.0])
    b = np.array([0.5, 0.85, 0.1])
    cs2 = deduplicator.cosine_similarity(a, b)
    print(cs2)

    # Different → similarity ≈ 0.0
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    cs3 = deduplicator.cosine_similarity(a, b)
    print(cs3)
    
    
    articles = [
      {
          "title": "Fed Raises Interest Rates by 0.25%",
          "description": "The Federal Reserve raised interest rates by a quarter point on Wednesday.",
          "url": "https://example.com/fed-rate-hike",
          "source": "Reuters"
      },
      {
          "title": "Federal Reserve Increases Rates by 25 Basis Points",
          "description": "The Fed hiked rates by 0.25% at its latest meeting.",
          "url": "https://example.com/fed-hike-2",
          "source": "CNBC"
      },
      {
          "title": "Apple Reports Record Q4 Earnings",
          "description": "Apple beat Wall Street expectations with strong iPhone sales.",
          "url": "https://example.com/apple-earnings",
          "source": "Yahoo Finance"
      },
      {
          "title": "Bitcoin Surges Past $100,000",
          "description": "Bitcoin hit a new all-time high driven by institutional demand.",
          "url": "https://example.com/bitcoin-100k",
          "source": "CNBC"
      },
      {
          "title": "BTC Breaks $100K Milestone",
          "description": "Bitcoin surpassed one hundred thousand dollars for the first time.",
          "url": "https://example.com/btc-milestone",
          "source": "Reuters"
      }
  ]
    da = deduplicator.deduplicate_articles(articles)
    print(da)
    
    