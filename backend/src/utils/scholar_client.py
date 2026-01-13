"""
Research Paper Retriever.

Integrates with Semantic Scholar API for academic paper retrieval.
Includes DOI resolution and citation analysis.
"""

import os
import time
import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class Author:
    """Research paper author."""
    name: str
    author_id: Optional[str] = None
    affiliations: List[str] = field(default_factory=list)


@dataclass
class Paper:
    """Research paper with metadata."""
    paper_id: str
    title: str
    abstract: str
    authors: List[Author]
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    citation_count: int = 0
    reference_count: int = 0
    influential_citation_count: int = 0
    fields_of_study: List[str] = field(default_factory=list)
    publication_types: List[str] = field(default_factory=list)
    tldr: Optional[str] = None
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": [{"name": a.name, "id": a.author_id} for a in self.authors],
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "url": self.url,
            "citation_count": self.citation_count,
            "fields_of_study": self.fields_of_study,
            "relevance_score": self.relevance_score
        }
    
    def to_text(self) -> str:
        """Convert to text for embedding."""
        parts = [f"Title: {self.title}"]
        if self.authors:
            parts.append(f"Authors: {', '.join(a.name for a in self.authors[:5])}")
        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")
        if self.venue:
            parts.append(f"Venue: {self.venue}")
        if self.year:
            parts.append(f"Year: {self.year}")
        return "\n".join(parts)


class SemanticScholarClient:
    """
    Semantic Scholar API client.
    
    Free API with rate limits (100 requests/5 min without key).
    API key available for higher limits.
    
    Usage:
        client = SemanticScholarClient()
        papers = client.search("diabetes machine learning", limit=20)
        
        # Get citations
        citations = client.get_citations("paper_id", limit=50)
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    PAPER_FIELDS = [
        "paperId", "title", "abstract", "year", "venue",
        "authors", "externalIds", "url", "openAccessPdf",
        "citationCount", "referenceCount", "influentialCitationCount",
        "fieldsOfStudy", "publicationTypes", "tldr"
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: float = 0.5
    ):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        self.rate_limit = rate_limit
        self._last_request = 0
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers["x-api-key"] = self.api_key
    
    def _rate_limit_wait(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()
    
    def _parse_paper(self, data: Dict, rank: int = 0, total: int = 1) -> Paper:
        """Parse API response to Paper object."""
        authors = []
        for a in data.get("authors", []):
            authors.append(Author(
                name=a.get("name", "Unknown"),
                author_id=a.get("authorId")
            ))
        
        external_ids = data.get("externalIds", {}) or {}
        
        return Paper(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            authors=authors,
            year=data.get("year"),
            venue=data.get("venue"),
            doi=external_ids.get("DOI"),
            arxiv_id=external_ids.get("ArXiv"),
            url=data.get("url"),
            pdf_url=data.get("openAccessPdf", {}).get("url") if data.get("openAccessPdf") else None,
            citation_count=data.get("citationCount", 0) or 0,
            reference_count=data.get("referenceCount", 0) or 0,
            influential_citation_count=data.get("influentialCitationCount", 0) or 0,
            fields_of_study=data.get("fieldsOfStudy", []) or [],
            publication_types=data.get("publicationTypes", []) or [],
            tldr=data.get("tldr", {}).get("text") if data.get("tldr") else None,
            relevance_score=1.0 - (rank / max(total, 1))
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def search(
        self,
        query: str,
        limit: int = 20,
        year_range: Optional[tuple] = None,
        fields_of_study: Optional[List[str]] = None,
        open_access_only: bool = False
    ) -> List[Paper]:
        """
        Search for papers.
        
        Args:
            query: Search query
            limit: Maximum results
            year_range: (start_year, end_year) tuple
            fields_of_study: Filter by fields (e.g., ["Medicine", "Biology"])
            open_access_only: Only return open access papers
        """
        self._rate_limit_wait()
        
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": ",".join(self.PAPER_FIELDS)
        }
        
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        
        if open_access_only:
            params["openAccessPdf"] = ""
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/paper/search",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for i, item in enumerate(data.get("data", [])):
                paper = self._parse_paper(item, i, limit)
                papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers for query: {query[:50]}...")
            return papers
            
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get paper by ID (Semantic Scholar ID, DOI, ArXiv ID, etc.)."""
        self._rate_limit_wait()
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/paper/{paper_id}",
                params={"fields": ",".join(self.PAPER_FIELDS)},
                timeout=30
            )
            response.raise_for_status()
            return self._parse_paper(response.json())
            
        except Exception as e:
            logger.error(f"Error fetching paper {paper_id}: {e}")
            return None
    
    def get_paper_by_doi(self, doi: str) -> Optional[Paper]:
        """Get paper by DOI."""
        return self.get_paper(f"DOI:{doi}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_citations(
        self,
        paper_id: str,
        limit: int = 50,
        fields: Optional[List[str]] = None
    ) -> List[Paper]:
        """Get papers that cite this paper."""
        self._rate_limit_wait()
        
        fields = fields or ["paperId", "title", "abstract", "year", "authors", "citationCount"]
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/paper/{paper_id}/citations",
                params={
                    "limit": min(limit, 1000),
                    "fields": ",".join(fields)
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for i, item in enumerate(data.get("data", [])):
                citing_paper = item.get("citingPaper", {})
                if citing_paper:
                    paper = self._parse_paper(citing_paper, i, limit)
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching citations for {paper_id}: {e}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_references(
        self,
        paper_id: str,
        limit: int = 50
    ) -> List[Paper]:
        """Get papers referenced by this paper."""
        self._rate_limit_wait()
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/paper/{paper_id}/references",
                params={
                    "limit": min(limit, 1000),
                    "fields": "paperId,title,abstract,year,authors,citationCount"
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for i, item in enumerate(data.get("data", [])):
                cited_paper = item.get("citedPaper", {})
                if cited_paper and cited_paper.get("paperId"):
                    paper = self._parse_paper(cited_paper, i, limit)
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching references for {paper_id}: {e}")
            return []
    
    def build_citation_network(
        self,
        seed_paper_id: str,
        depth: int = 1,
        max_citations: int = 20
    ) -> Dict[str, Any]:
        """
        Build citation network from a seed paper.
        
        Returns graph structure with papers and citation relationships.
        """
        visited: Set[str] = set()
        papers: Dict[str, Paper] = {}
        edges: List[tuple] = []
        
        def explore(paper_id: str, current_depth: int):
            if paper_id in visited or current_depth > depth:
                return
            
            visited.add(paper_id)
            
            # Get the paper
            paper = self.get_paper(paper_id)
            if paper:
                papers[paper_id] = paper
            
            if current_depth < depth:
                # Get citations
                citations = self.get_citations(paper_id, limit=max_citations)
                for citing_paper in citations:
                    if citing_paper.paper_id:
                        edges.append((citing_paper.paper_id, paper_id))
                        papers[citing_paper.paper_id] = citing_paper
                        explore(citing_paper.paper_id, current_depth + 1)
        
        explore(seed_paper_id, 0)
        
        return {
            "papers": {pid: p.to_dict() for pid, p in papers.items()},
            "edges": edges,
            "total_papers": len(papers),
            "total_edges": len(edges)
        }
    
    def search_medical(
        self,
        query: str,
        limit: int = 20,
        recent_years: int = 5
    ) -> List[Paper]:
        """Search specifically for medical/biomedical papers."""
        current_year = datetime.now().year
        
        return self.search(
            query=query,
            limit=limit,
            year_range=(current_year - recent_years, current_year),
            fields_of_study=["Medicine", "Biology"]
        )
