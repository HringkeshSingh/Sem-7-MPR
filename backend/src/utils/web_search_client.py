"""
Web Search Client.

Provides unified interface for web search APIs:
- Google Custom Search API
- Bing Web Search API
- DuckDuckGo (no API key required)
"""

import os
import time
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
import logging
import aiohttp
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Standardized search result."""
    title: str
    url: str
    snippet: str
    source: str  # google, bing, duckduckgo
    published_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0


class BaseSearchClient(ABC):
    """Abstract base for search clients."""
    
    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self._last_request = 0
    
    def _rate_limit_wait(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()
    
    @abstractmethod
    def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        pass
    
    @abstractmethod
    async def search_async(self, query: str, num_results: int = 10) -> List[SearchResult]:
        pass


class GoogleSearchClient(BaseSearchClient):
    """Google Custom Search API client."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cx: Optional[str] = None,
        rate_limit: float = 1.0
    ):
        super().__init__(rate_limit)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.cx = cx or os.getenv("GOOGLE_CX", "")  # Custom Search Engine ID
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search using Google Custom Search API."""
        if not self.api_key or not self.cx:
            logger.warning("Google API key or CX not configured")
            return []
        
        self._rate_limit_wait()
        results = []
        
        # Google CSE returns max 10 results per request
        for start in range(1, min(num_results + 1, 101), 10):
            try:
                params = {
                    "key": self.api_key,
                    "cx": self.cx,
                    "q": query,
                    "start": start,
                    "num": min(10, num_results - len(results))
                }
                
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                for item in data.get("items", []):
                    result = SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source="google",
                        metadata={
                            "display_link": item.get("displayLink", ""),
                            "mime_type": item.get("mime", ""),
                            "file_format": item.get("fileFormat", "")
                        },
                        relevance_score=1.0 - (len(results) / max(num_results, 1))
                    )
                    results.append(result)
                    
                    if len(results) >= num_results:
                        break
                        
            except Exception as e:
                logger.error(f"Google search error: {e}")
                break
        
        return results
    
    async def search_async(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Async Google search."""
        return await asyncio.to_thread(self.search, query, num_results)


class BingSearchClient(BaseSearchClient):
    """Bing Web Search API client."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: float = 1.0
    ):
        super().__init__(rate_limit)
        self.api_key = api_key or os.getenv("BING_API_KEY", "")
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search using Bing Web Search API."""
        if not self.api_key:
            logger.warning("Bing API key not configured")
            return []
        
        self._rate_limit_wait()
        
        try:
            headers = {"Ocp-Apim-Subscription-Key": self.api_key}
            params = {
                "q": query,
                "count": min(num_results, 50),
                "mkt": "en-US",
                "freshness": "Month"  # Recent results
            }
            
            response = requests.get(
                self.base_url,
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for i, item in enumerate(data.get("webPages", {}).get("value", [])):
                # Parse date if available
                published_date = None
                if item.get("dateLastCrawled"):
                    try:
                        published_date = datetime.fromisoformat(
                            item["dateLastCrawled"].replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass
                
                result = SearchResult(
                    title=item.get("name", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    source="bing",
                    published_date=published_date,
                    metadata={
                        "display_url": item.get("displayUrl", ""),
                        "is_navigational": item.get("isNavigational", False),
                        "language": item.get("language", "")
                    },
                    relevance_score=1.0 - (i / max(num_results, 1))
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Bing search error: {e}")
            return []
    
    async def search_async(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Async Bing search."""
        return await asyncio.to_thread(self.search, query, num_results)


class DuckDuckGoClient(BaseSearchClient):
    """DuckDuckGo search client (no API key required)."""
    
    def __init__(self, rate_limit: float = 2.0):
        super().__init__(rate_limit)
        self.base_url = "https://html.duckduckgo.com/html/"
    
    def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search using DuckDuckGo HTML endpoint."""
        self._rate_limit_wait()
        
        try:
            from bs4 import BeautifulSoup
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.post(
                self.base_url,
                data={"q": query},
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            results = []
            
            for i, result_div in enumerate(soup.select(".result")):
                if i >= num_results:
                    break
                
                title_elem = result_div.select_one(".result__title a")
                snippet_elem = result_div.select_one(".result__snippet")
                
                if title_elem:
                    # DuckDuckGo uses redirects
                    url = title_elem.get("href", "")
                    if "uddg=" in url:
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                        url = parsed.get("uddg", [url])[0]
                    
                    result = SearchResult(
                        title=title_elem.get_text(strip=True),
                        url=url,
                        snippet=snippet_elem.get_text(strip=True) if snippet_elem else "",
                        source="duckduckgo",
                        relevance_score=1.0 - (i / max(num_results, 1))
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    async def search_async(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Async DuckDuckGo search."""
        return await asyncio.to_thread(self.search, query, num_results)


class WebSearchClient:
    """
    Unified web search client with multiple providers.
    
    Usage:
        client = WebSearchClient()
        results = client.search("diabetes treatment guidelines", num_results=20)
        
        # Medical-specific search
        results = client.search_medical("COVID-19 vaccine efficacy")
    """
    
    MEDICAL_SITES = [
        "nih.gov", "ncbi.nlm.nih.gov", "who.int", "cdc.gov",
        "mayoclinic.org", "webmd.com", "medscape.com",
        "nejm.org", "thelancet.com", "jamanetwork.com",
        "bmj.com", "nature.com/nm", "sciencedirect.com"
    ]
    
    def __init__(
        self,
        google_api_key: Optional[str] = None,
        google_cx: Optional[str] = None,
        bing_api_key: Optional[str] = None,
        preferred_provider: str = "duckduckgo"
    ):
        self.providers = {
            "google": GoogleSearchClient(google_api_key, google_cx),
            "bing": BingSearchClient(bing_api_key),
            "duckduckgo": DuckDuckGoClient()
        }
        self.preferred_provider = preferred_provider
    
    def search(
        self,
        query: str,
        num_results: int = 10,
        provider: Optional[str] = None
    ) -> List[SearchResult]:
        """Search using specified or preferred provider."""
        provider_name = provider or self.preferred_provider
        client = self.providers.get(provider_name)
        
        if client is None:
            logger.warning(f"Unknown provider: {provider_name}, using duckduckgo")
            client = self.providers["duckduckgo"]
        
        return client.search(query, num_results)
    
    def search_medical(
        self,
        query: str,
        num_results: int = 10,
        provider: Optional[str] = None
    ) -> List[SearchResult]:
        """Search with medical site restrictions."""
        # Add site restrictions for medical sources
        site_query = " OR ".join([f"site:{site}" for site in self.MEDICAL_SITES[:5]])
        medical_query = f"{query} ({site_query})"
        
        return self.search(medical_query, num_results, provider)
    
    async def search_all_providers(
        self,
        query: str,
        num_results_per_provider: int = 5
    ) -> List[SearchResult]:
        """Search all available providers concurrently."""
        tasks = []
        for name, client in self.providers.items():
            tasks.append(client.search_async(query, num_results_per_provider))
        
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        combined = []
        for results in all_results:
            if isinstance(results, list):
                combined.extend(results)
        
        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in combined:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        available = []
        
        if self.providers["google"].api_key and self.providers["google"].cx:
            available.append("google")
        if self.providers["bing"].api_key:
            available.append("bing")
        available.append("duckduckgo")  # Always available
        
        return available
