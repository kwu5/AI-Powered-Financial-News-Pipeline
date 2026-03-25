import feedparser
import logging
logger = logging.getLogger(__name__)
from datetime import datetime

from typing import List


FEED_URLS = [
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",  # CNBC
    "https://finance.yahoo.com/news/rssindex",                                              # Yahoo Finance
    "https://ir.thomsonreuters.com/rss/news-releases.xml?items=15",                         # Reuters Business
]

class RSSREADER:
    def __init__(self):
        self.feed_urls = FEED_URLS
    
    def fetch_from_feeds(self) -> List[dict]:
        all_articles = []
        for url in self.feed_urls:
            try:
                feed = feedparser.parse(url)
                
                for entry in feed.entries:
                    article = {
                        "title": entry.get("title", ""),
                        "description": entry.get("summary", ""),
                        "content": entry.get("summary", ""),  # RSS usually only has summary
                        "url": entry.get("link", ""),
                        "source": feed.get("title", url),
                        "published_at": self._parse_date(entry),
                        "fetched_at": datetime.now().isoformat(),
                    }
                    if article["title"] and article["url"]:
                        all_articles.append(article)
            except Exception as e:
                logger.error(f"Failed to fetch feed {url} : {e}")
                continue   
        return all_articles
    
    def _parse_date(self, entry) -> str:
        """Convert feedparser's parsed date to ISO format string."""
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            return datetime(*entry.published_parsed[:6]).isoformat()
        return datetime.now().isoformat()


if __name__ == '__main__':
    rssreder = RSSREADER()
    result = rssreder.fetch_from_feeds()
    print(result)
    print("-------------------------")
    for a in result[:3]:
        print(f"{a['source']} | {a['title']}")
    
    