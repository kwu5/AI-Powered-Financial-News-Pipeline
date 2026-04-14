

from typing import List
from sentence_transformers import SentenceTransformer
from src.config import Settings
import numpy as np

#Converts text into numerical vectors (arrays of numbers)
class EmbeddingGenerator:
    def __init__(self, settings: Settings) -> None:
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        

    def generate_embedding(self, text: str) -> np.ndarray:
        embedding = np.asarray(self.model.encode(text, normalize_embeddings=True))
        return embedding
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = np.asarray(self.model.encode(texts, normalize_embeddings=True))
        return embeddings



if __name__ == '__main__':
    embeddingGenerator = EmbeddingGenerator(Settings())     # type: ignore
    res1 = embeddingGenerator.generate_embedding('Test1')
    res2 = embeddingGenerator.generate_embeddings(['This', 'is', 'Test2'])
    print(res1)
    print(res2)
    
    
    
    
    