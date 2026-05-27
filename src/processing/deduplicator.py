
import logging
from typing import Dict, List
import numpy as np
from src.config import Settings
from src.processing.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class Deduplicator:
    def __init__(self, embedding_generator: EmbeddingGenerator) -> None:
        self.embedding_generator = embedding_generator
        self.settings = Settings()      # type: ignore

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _dedup_by_key(self, articles: List[Dict], key: str) -> List[Dict]:
        """Keep the first article seen for each unique value of `articles[i][key]`."""
        seen = set()
        kept = []
        for a in articles:
            v = a[key]
            if v in seen:
                continue
            seen.add(v)
            kept.append(a)
        return kept

    def _dedup_by_similarity(self, articles: List[Dict]) -> List[Dict]:
        """Drop articles whose title+description embedding is too close to a kept one."""
        kept: List[Dict] = []
        kept_embeddings: List[np.ndarray] = []
        for a in articles:
            text = a["title"] + " " + a.get("description", "")
            emb = self.embedding_generator.generate_embedding(text)
            if any(
                self.cosine_similarity(emb, e) >= self.settings.SIMILARITY_THRESHOLD
                for e in kept_embeddings
            ):
                continue
            kept.append(a)
            kept_embeddings.append(emb)
        return kept

    def deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Run all three dedup stages and log per-stage drops.

        Stage 1: canonical_url (exact match)
        Stage 2: content_hash (sha256 of normalized title+content)
        Stage 3: cosine similarity of title+description embeddings
        """
        n0 = len(articles)
        after_url = self._dedup_by_key(articles, "canonical_url")
        n1 = len(after_url)
        after_hash = self._dedup_by_key(after_url, "content_hash")
        n2 = len(after_hash)
        after_sim = self._dedup_by_similarity(after_hash)
        n3 = len(after_sim)

        logger.info(
            f"Dedup: {n0} -> {n1} (url) -> {n2} (hash) -> {n3} (sim) "
            f"[dropped {n0 - n1}/{n1 - n2}/{n2 - n3}]"
        )
        return after_sim
        
    
if __name__ == '__main__':
    embeddingGenerator = EmbeddingGenerator(Settings())     # type: ignore
    deduplicator = Deduplicator(embeddingGenerator)
    # Identical direction -> similarity ≈ 1.0
    a = np.array([0.6, 0.8, 0.0])
    b = np.array([0.6, 0.8, 0.0])
    cs1 = deduplicator.cosine_similarity(a, b)
    print(cs1)

    # Similar -> similarity ≈ 0.9+
    a = np.array([0.6, 0.8, 0.0])
    b = np.array([0.5, 0.85, 0.1])
    cs2 = deduplicator.cosine_similarity(a, b)
    print(cs2)

    # Different -> similarity ≈ 0.0
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
    
    