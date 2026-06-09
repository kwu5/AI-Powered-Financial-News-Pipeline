
from typing import Dict, List
from src.config import Settings
from openai import OpenAI
from pydantic import BaseModel


class GroundedLLMResponse(BaseModel):
    """Flat structured-output schema for the grounded answer call.

    Kept flat (no nested objects) because gpt-4o-mini's structured output is
    most reliable that way. qa.py reads these three fields off the parsed result;
    it builds the authoritative Citation list from the retrieved hits, not from
    here — `used_markers` only tells us WHICH numbered sources the model cited.
    """
    answer: str                   # may contain inline [n] markers
    used_markers: List[int]       # 1-based source numbers the model cited
    answered_from_context: bool   # False -> context didn't contain the answer


class LLMClient:
    def __init__(self,setting: Settings):
        self.client = OpenAI(api_key= setting.OPENAI_API_KEY)
        self.model = setting.LLM_MODEL
    
    def generate_summary(self, articles: List[Dict]) -> None|str:
        formatted = []
        for i,a in enumerate(articles,1):
            preview = a["content"][:500]
            formatted.append(
                f"[Article {i}]\n"
                f"Title: {a['title']}\n"
                f"Source: {a['source']}\n"
                f"Content: {preview}...\n"
            )
        system_prompt = """You are a financial news analyst. Summarize the provided articles into a
                structured daily briefing with these exact sections:
                ## Major Market Movements
                ## Federal Reserve & Monetary Policy
                ## Corporate Earnings & News
                ## Cryptocurrency & Digital Assets
                ## Key Themes of the Day
                ## Market Sentiment

                Use bullet points under each section. Cite sources inline (e.g., "per Reuters"). If a section has no
                relevant news, write "No significant updates." Be concise and factual."""
        
        user_prompt = f"Summarize today's financial news:\n\n{formatted}"
        response = self.client.chat.completions.create(
            model = self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3, # adjusts how the model picks the next token from its probability distribution.
            max_tokens=2000
        )
        return response.choices[0].message.content 

    def generate_grounded_answer(self, query: str, numbered_context: str) -> GroundedLLMResponse:
        """Answer `query` using ONLY `numbered_context`, citing sources by number.

        `numbered_context` is the "[1] <chunk>\\n\\n[2] <chunk>..." block qa.py
        builds from the retrieved hits. The model must answer strictly from it,
        cite each claim with the bracketed source number, and flag when the
        context is insufficient instead of falling back on outside knowledge.
        temperature=0 so Ships F/G can replay the same queries deterministically.
        """
        system_prompt = (
            "You are a financial-news question answerer. Answer the user's "
            "question using ONLY the numbered sources provided in the context. "
            "Rules:\n"
            "- Cite every claim with the bracketed source number it came from, "
            "e.g. 'The Fed held rates [2].' Use the numbers exactly as given.\n"
            "- List every source number you relied on in `used_markers`.\n"
            "- If the context does not contain enough information to answer, set "
            "`answered_from_context` to false and say you don't have enough "
            "indexed context — do NOT use outside knowledge or guess.\n"
            "- Otherwise set `answered_from_context` to true."
        )
        user_prompt = f"Question: {query}\n\nContext:\n{numbered_context}"
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format=GroundedLLMResponse,
        )
        parsed = completion.choices[0].message.parsed
        # parse() can return None if the model refused; surface a safe insufficient-context
        # result rather than letting qa.py dereference None.
        if parsed is None:
            return GroundedLLMResponse(
                answer="I don't have enough indexed context to answer that.",
                used_markers=[],
                answered_from_context=False,
            )
        return parsed

    def classify_sentiment(self, text: str) -> str:
        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
            {"role": "system", "content": "Classify the sentiment of the following financial text. Respond with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL."},
            {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=5,  
        )
        content = response.choices[0].message.content or ""
        return content.strip().upper()
        
        

if __name__ == "__main__":
    settings = Settings()  # type: ignore
    llm_client = LLMClient(settings)

    fake_articles = [
          {
              "title": "Fed Raises Interest Rates by 0.25%",
              "source": "Reuters",
              "content": "The Federal Reserve raised interest rates by 25 basis points on Wednesday,signaling a cautious approach to taming inflation. Chair Powell hinted at one more hike before year-end.",
          },
          {
              "title": "Apple Reports Record Q2 Earnings",
              "source": "CNBC",
              "content": "Apple Inc reported record quarterly revenue of $95 billion, driven by strong iPhone 16 sales and services growth. Shares jumped 4% in after-hours trading.",
          },
          {
              "title": "Bitcoin Surges Past $70,000",
              "source": "Yahoo Finance",
              "content": "Bitcoin surged past $70,000 for the first time this quarter as institutional investors increased their holdings following spot ETF inflows.",
          },
      ]

    print("=== Testing generate_summary ===")
    summary = llm_client.generate_summary(fake_articles)
    print(summary)

    print("\n=== Testing classify_sentiment ===")
    samples = [
          "Apple reported record earnings and shares jumped 4%.",
          "Tesla shares plunged 8% after missing delivery targets.",
          "The S&P 500 closed flat as traders awaited the Fed decision.",
      ]
    for text in samples:
        sentiment = llm_client.classify_sentiment(text)
        print(f"[{sentiment}] {text}")

        