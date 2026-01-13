"""
Medical News Aggregator.

Aggregates medical news from:
- NewsAPI with medical filters
- RSS feeds from major medical journals
- Medical news sites
"""

import os
import time
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import requests
import feedparser
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Medical news article."""
    title: str
    description: str
    url: str
    source: str
    published_date: Optional[datetime] = None
    author: Optional[str] = None
    image_url: Optional[str] = None
    category: str = "general"
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "url": self.url,
            "source": self.source,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "author": self.author,
            "category": self.category,
            "relevance_score": self.relevance_score
        }
    
    def to_text(self) -> str:
        return f"Title: {self.title}\nSource: {self.source}\nContent: {self.description}"


# RSS feeds for major medical journals and news sources
MEDICAL_RSS_FEEDS = {
    "nejm": {
        "url": "https://www.nejm.org/action/showFeed?jc=nejm&type=etoc&feed=rss",
        "name": "New England Journal of Medicine",
        "category": "research"
    },
    "lancet": {
        "url": "https://www.thelancet.com/rssfeed/lancet_current.xml",
        "name": "The Lancet",
        "category": "research"
    },
    "jama": {
        "url": "https://jamanetwork.com/rss/site_3/67.xml",
        "name": "JAMA Network",
        "category": "research"
    },
    "bmj": {
        "url": "https://www.bmj.com/rss/recent.xml",
        "name": "British Medical Journal",
        "category": "research"
    },
    "nature_medicine": {
        "url": "https://www.nature.com/nm.rss",
        "name": "Nature Medicine",
        "category": "research"
    },
    "cdc_health": {
        "url": "https://tools.cdc.gov/api/v2/resources/media/404952.rss",
        "name": "CDC Health Updates",
        "category": "public_health"
    },
    "who_news": {
        "url": "https://www.who.int/rss-feeds/news-english.xml",
        "name": "WHO News",
        "category": "public_health"
    },
    "nih_news": {
        "url": "https://www.nih.gov/news-events/news-releases/feed.xml",
        "name": "NIH News",
        "category": "research"
    },
    "medscape": {
        "url": "https://www.medscape.com/cx/rssfeeds/2701.xml",
        "name": "Medscape Medical News",
        "category": "clinical"
    },
    "medical_news_today": {
        "url": "https://www.medicalnewstoday.com/rss/medical_news.xml",
        "name": "Medical News Today",
        "category": "general"
    }
}


class NewsAPIClient:
    """NewsAPI client for medical news."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWS_API_KEY", "")
        self.base_url = "https://newsapi.org/v2"
        self._last_request = 0
        self.rate_limit = 1.0
    
    def _rate_limit_wait(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def search(
        self,
        query: str,
        max_results: int = 20,
        days_back: int = 7,
        language: str = "en"
    ) -> List[NewsArticle]:
        """Search for news articles."""
        if not self.api_key:
            logger.warning("NewsAPI key not configured")
            return []
        
        self._rate_limit_wait()
        
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        try:
            params = {
                "apiKey": self.api_key,
                "q": query,
                "from": from_date,
                "language": language,
                "sortBy": "relevancy",
                "pageSize": min(max_results, 100)
            }
            
            response = requests.get(
                f"{self.base_url}/everything",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for i, item in enumerate(data.get("articles", [])):
                # Parse date
                pub_date = None
                if item.get("publishedAt"):
                    try:
                        pub_date = datetime.fromisoformat(
                            item["publishedAt"].replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass
                
                article = NewsArticle(
                    title=item.get("title", ""),
                    description=item.get("description", "") or item.get("content", ""),
                    url=item.get("url", ""),
                    source=item.get("source", {}).get("name", "Unknown"),
                    published_date=pub_date,
                    author=item.get("author"),
                    image_url=item.get("urlToImage"),
                    category="general",
                    relevance_score=1.0 - (i / max(max_results, 1))
                )
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return []
    
    def search_medical(
        self,
        topic: str,
        max_results: int = 20
    ) -> List[NewsArticle]:
        """Search for medical news with health domain focus."""
        # Add medical context to query
        medical_query = f"{topic} AND (health OR medical OR clinical OR treatment OR disease)"
        return self.search(medical_query, max_results)


class RSSFeedClient:
    """RSS feed parser for medical journals."""
    
    def __init__(self, feeds: Optional[Dict] = None):
        self.feeds = feeds or MEDICAL_RSS_FEEDS
    
    def fetch_feed(self, feed_key: str, max_items: int = 20) -> List[NewsArticle]:
        """Fetch articles from a specific RSS feed."""
        if feed_key not in self.feeds:
            logger.warning(f"Unknown feed: {feed_key}")
            return []
        
        feed_config = self.feeds[feed_key]
        
        try:
            parsed = feedparser.parse(feed_config["url"])
            
            articles = []
            for i, entry in enumerate(parsed.entries[:max_items]):
                # Parse date
                pub_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    pub_date = datetime(*entry.updated_parsed[:6])
                
                # Get description
                description = ""
                if hasattr(entry, 'summary'):
                    description = entry.summary
                elif hasattr(entry, 'description'):
                    description = entry.description
                
                # Clean HTML from description
                from bs4 import BeautifulSoup
                if description:
                    description = BeautifulSoup(description, "lxml").get_text()[:1000]
                
                article = NewsArticle(
                    title=entry.get("title", ""),
                    description=description,
                    url=entry.get("link", ""),
                    source=feed_config["name"],
                    published_date=pub_date,
                    author=entry.get("author"),
                    category=feed_config.get("category", "general"),
                    relevance_score=1.0 - (i / max(max_items, 1))
                )
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from {feed_config['name']}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching RSS feed {feed_key}: {e}")
            return []
    
    def fetch_all_feeds(
        self,
        max_per_feed: int = 10,
        categories: Optional[List[str]] = None
    ) -> List[NewsArticle]:
        """Fetch articles from all configured feeds."""
        all_articles = []
        
        for feed_key, feed_config in self.feeds.items():
            if categories and feed_config.get("category") not in categories:
                continue
            
            articles = self.fetch_feed(feed_key, max_per_feed)
            all_articles.extend(articles)
        
        # Sort by date (most recent first)
        all_articles.sort(
            key=lambda x: x.published_date or datetime.min,
            reverse=True
        )
        
        return all_articles
    
    async def fetch_all_feeds_async(
        self,
        max_per_feed: int = 10
    ) -> List[NewsArticle]:
        """Fetch all feeds concurrently."""
        tasks = []
        for feed_key in self.feeds:
            tasks.append(asyncio.to_thread(self.fetch_feed, feed_key, max_per_feed))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_articles = []
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
        
        return all_articles


class MedicalNewsAggregator:
    """
    Unified medical news aggregator.
    
    Usage:
        aggregator = MedicalNewsAggregator()
        
        # Get latest from all sources
        news = aggregator.get_latest(max_per_source=20)
        
        # Search specific topic
        news = aggregator.search("COVID-19 vaccine", max_results=50)
    """
    
    def __init__(
        self,
        news_api_key: Optional[str] = None,
        custom_feeds: Optional[Dict] = None
    ):
        self.news_api = NewsAPIClient(news_api_key)
        self.rss_client = RSSFeedClient(custom_feeds)
    
    def get_latest(
        self,
        max_per_source: int = 10,
        categories: Optional[List[str]] = None,
        include_news_api: bool = True
    ) -> List[NewsArticle]:
        """Get latest medical news from all sources."""
        all_articles = []
        
        # Get from RSS feeds
        rss_articles = self.rss_client.fetch_all_feeds(max_per_source, categories)
        all_articles.extend(rss_articles)
        
        # Get from NewsAPI if configured
        if include_news_api and self.news_api.api_key:
            news_articles = self.news_api.search_medical("health medicine", max_per_source * 2)
            all_articles.extend(news_articles)
        
        # Deduplicate by URL
        seen = set()
        unique = []
        for article in all_articles:
            if article.url not in seen:
                seen.add(article.url)
                unique.append(article)
        
        return unique
    
    def search(
        self,
        query: str,
        max_results: int = 50,
        use_news_api: bool = True
    ) -> List[NewsArticle]:
        """Search for medical news on a topic."""
        articles = []
        
        if use_news_api and self.news_api.api_key:
            api_results = self.news_api.search_medical(query, max_results)
            articles.extend(api_results)
        
        # RSS feeds don't support search, but we can filter by topic
        rss_articles = self.rss_client.fetch_all_feeds(max_per_feed=20)
        
        query_lower = query.lower()
        for article in rss_articles:
            text = f"{article.title} {article.description}".lower()
            if any(term in text for term in query_lower.split()):
                articles.append(article)
        
        return articles[:max_results]
    
    def get_by_category(
        self,
        category: str,
        max_results: int = 20
    ) -> List[NewsArticle]:
        """Get news by category (research, clinical, public_health, general)."""
        return self.rss_client.fetch_all_feeds(
            max_per_feed=max_results,
            categories=[category]
        )
    
    def get_journal_articles(
        self,
        journal_keys: Optional[List[str]] = None,
        max_per_journal: int = 10
    ) -> List[NewsArticle]:
        """Get articles from specific journals."""
        if journal_keys is None:
            journal_keys = ["nejm", "lancet", "jama", "bmj", "nature_medicine"]
        
        articles = []
        for key in journal_keys:
            journal_articles = self.rss_client.fetch_feed(key, max_per_journal)
            articles.extend(journal_articles)
        
        return articles
