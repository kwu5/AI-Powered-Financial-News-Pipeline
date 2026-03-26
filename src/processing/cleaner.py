import spacy
import re


class TextCleaner:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")        
        
    
    def clean_article(self, text: str) -> str:
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        # Remove common ad phrases
        ad_phrases = [
            "subscribe now", "click here", "sign up for",
            "advertisement", "sponsored content",
        ]
        for phrase in ad_phrases:
            text = re.sub(re.escape(phrase), '', text, flags=re.IGNORECASE)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text      
        

    def extract_entities(self, text: str) -> dict:
        doc = self.nlp(text)
        entities = {"ORG":[], "PERSON":[], "MONEY":[], "PERCENT":[]}
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        return entities
    
if __name__ == '__main__':
    cleaner = TextCleaner()

    sample = "Apple Inc. CEO Tim Cook announced https://example.com earnings of $1.2 billion, up 15%. Subscribe now for more news. Contact info@test.com"

    cleaned = cleaner.clean_article(sample)
    print("Cleaned:", cleaned)

    entities = cleaner.extract_entities(cleaned)
    print("Entities:", entities)