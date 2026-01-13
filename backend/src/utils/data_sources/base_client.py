"""
Base client interface for all data source clients.

All data source clients should inherit from this base class
to ensure consistent interface across different sources.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """Standard document format from any data source."""
    source: str  # e.g., 'pubmed', 'clinical_trials', 'who'
    source_id: str  # Unique ID within source
    title: str
    content: str
    url: Optional[str] = None
    authors: Optional[List[str]] = None
    publication_date: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'source': self.source,
            'source_id': self.source_id,
            'title': self.title,
            'content': self.content,
            'url': self.url,
            'authors': self.authors or [],
            'publication_date': self.publication_date.isoformat() if self.publication_date else None,
            'metadata': self.metadata or {},
            'relevance_score': self.relevance_score
        }
    
    def to_text(self) -> str:
        """Convert to text for embedding."""
        parts = [f"Title: {self.title}"]
        if self.authors:
            parts.append(f"Authors: {', '.join(self.authors[:3])}")
        parts.append(f"Content: {self.content}")
        if self.metadata:
            for key, value in self.metadata.items():
                if isinstance(value, str) and len(value) < 100:
                    parts.append(f"{key}: {value}")
        return "\n".join(parts)


class BaseDataSourceClient(ABC):
    """
    Abstract base class for data source clients.
    
    All data source clients must implement:
    - search(): Search for documents matching a query
    - fetch_details(): Fetch detailed information for specific IDs
    - is_available(): Check if the source is available
    """
    
    def __init__(self, source_name: str, rate_limit: float = 1.0):
        """
        Initialize base client.
        
        Args:
            source_name: Name of the data source
            rate_limit: Minimum seconds between requests
        """
        self.source_name = source_name
        self.rate_limit = rate_limit
        self._last_request_time = 0
        self._is_available = None
        
    @abstractmethod
    def search(self, query: str, max_results: int = 20) -> List[RetrievedDocument]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of RetrievedDocument objects
        """
        pass
    
    @abstractmethod
    def fetch_by_id(self, doc_id: str) -> Optional[RetrievedDocument]:
        """
        Fetch a specific document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            RetrievedDocument or None if not found
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if the data source is available.
        
        Returns:
            True if source is accessible
        """
        if self._is_available is None:
            self._is_available = self._check_availability()
        return self._is_available
    
    @abstractmethod
    def _check_availability(self) -> bool:
        """Check if the source is accessible."""
        pass
    
    def build_healthcare_query(
        self,
        conditions: Optional[List[str]] = None,
        demographics: Optional[Dict[str, Any]] = None,
        keywords: Optional[List[str]] = None
    ) -> str:
        """
        Build an optimized query for healthcare data retrieval.
        
        Args:
            conditions: List of medical conditions
            demographics: Patient demographics
            keywords: Additional keywords
            
        Returns:
            Formatted query string
        """
        query_parts = []
        
        if conditions:
            query_parts.extend(conditions)
        
        if demographics:
            if demographics.get('age_range'):
                min_age, max_age = demographics['age_range']
                if min_age >= 65:
                    query_parts.append('elderly')
                elif max_age <= 18:
                    query_parts.append('pediatric')
            if demographics.get('gender'):
                query_parts.append(demographics['gender'])
        
        if keywords:
            query_parts.extend(keywords)
        
        return ' '.join(query_parts)
    
    def _rate_limit_wait(self):
        """Implement rate limiting."""
        import time
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()
