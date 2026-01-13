"""
Enhanced PubMed Client.

Extends the base PubMed client with:
- Bulk retrieval capabilities
- Relevance scoring based on medical ontologies
- Citation network analysis
- MeSH term expansion
"""

import os
import time
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import xml.etree.ElementTree as ET
import logging
import requests
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.pubmed_client import PubMedClient, PubMedArticle

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPubMedArticle(PubMedArticle):
    """Extended PubMed article with additional metadata."""
    mesh_terms: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    publication_types: List[str] = field(default_factory=list)
    cited_by_count: int = 0
    references: List[str] = field(default_factory=list)  # List of PMIDs
    pmc_id: Optional[str] = None
    full_text_url: Optional[str] = None
    funding_sources: List[str] = field(default_factory=list)
    affiliations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "doi": self.doi,
            "mesh_terms": self.mesh_terms,
            "keywords": self.keywords,
            "publication_types": self.publication_types,
            "cited_by_count": self.cited_by_count,
            "pmc_id": self.pmc_id,
            "relevance_score": self.relevance_score
        }
    
    def to_text(self) -> str:
        """Convert to text for embedding."""
        parts = [f"Title: {self.title}"]
        if self.authors:
            parts.append(f"Authors: {', '.join(self.authors[:5])}")
        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")
        if self.mesh_terms:
            parts.append(f"MeSH Terms: {', '.join(self.mesh_terms[:10])}")
        if self.journal:
            parts.append(f"Journal: {self.journal}")
        return "\n".join(parts)


# Medical ontology for relevance scoring
MEDICAL_ONTOLOGY = {
    "DIABETES": {
        "mesh_terms": ["Diabetes Mellitus", "Diabetes Mellitus, Type 2", "Diabetes Mellitus, Type 1",
                       "Diabetic Nephropathy", "Diabetic Retinopathy", "Hyperglycemia"],
        "keywords": ["insulin", "glucose", "hba1c", "metformin", "glycemic"],
        "weight": 1.0
    },
    "CARDIOVASCULAR": {
        "mesh_terms": ["Cardiovascular Diseases", "Heart Failure", "Myocardial Infarction",
                       "Coronary Artery Disease", "Atrial Fibrillation", "Hypertension"],
        "keywords": ["cardiac", "heart", "coronary", "angina", "arrhythmia"],
        "weight": 1.0
    },
    "RESPIRATORY": {
        "mesh_terms": ["Respiratory Tract Diseases", "Pulmonary Disease, Chronic Obstructive",
                       "Asthma", "Pneumonia", "Respiratory Insufficiency"],
        "keywords": ["lung", "pulmonary", "copd", "ventilation", "bronchial"],
        "weight": 1.0
    },
    "ONCOLOGY": {
        "mesh_terms": ["Neoplasms", "Carcinoma", "Tumor", "Cancer", "Metastasis"],
        "keywords": ["chemotherapy", "radiation", "oncology", "tumor", "malignant"],
        "weight": 1.0
    },
    "NEUROLOGY": {
        "mesh_terms": ["Nervous System Diseases", "Stroke", "Alzheimer Disease",
                       "Parkinson Disease", "Epilepsy", "Multiple Sclerosis"],
        "keywords": ["brain", "neurological", "cognitive", "seizure", "dementia"],
        "weight": 1.0
    },
    "INFECTIOUS": {
        "mesh_terms": ["Infection", "Sepsis", "Bacterial Infections", "Viral Infections"],
        "keywords": ["infection", "bacterial", "viral", "antibiotic", "pathogen"],
        "weight": 1.0
    }
}


class EnhancedPubMedClient(PubMedClient):
    """
    Enhanced PubMed client with advanced features.
    
    Usage:
        client = EnhancedPubMedClient()
        
        # Bulk search with pagination
        articles = client.bulk_search("diabetes treatment", max_results=500)
        
        # Get citation network
        network = client.get_citation_network("12345678", depth=2)
        
        # Score articles by medical relevance
        scored = client.score_articles(articles, ["DIABETES", "CARDIOVASCULAR"])
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._executor = ThreadPoolExecutor(max_workers=3)
    
    def _parse_enhanced_article(self, article_elem: ET.Element) -> Optional[EnhancedPubMedArticle]:
        """Parse XML for enhanced article data."""
        try:
            # Get base article data
            base = self._parse_article_xml(article_elem)
            if not base:
                return None
            
            # Extract MeSH terms
            mesh_terms = []
            for mesh in article_elem.findall('.//MeshHeading/DescriptorName'):
                if mesh.text:
                    mesh_terms.append(mesh.text)
            
            # Extract keywords
            keywords = []
            for kw in article_elem.findall('.//Keyword'):
                if kw.text:
                    keywords.append(kw.text)
            
            # Extract publication types
            pub_types = []
            for pt in article_elem.findall('.//PublicationType'):
                if pt.text:
                    pub_types.append(pt.text)
            
            # Extract PMC ID if available
            pmc_id = None
            pmc_elem = article_elem.find('.//ArticleId[@IdType="pmc"]')
            if pmc_elem is not None and pmc_elem.text:
                pmc_id = pmc_elem.text
            
            # Extract affiliations
            affiliations = []
            for aff in article_elem.findall('.//Affiliation'):
                if aff.text:
                    affiliations.append(aff.text)
            
            # Build full text URL if PMC available
            full_text_url = None
            if pmc_id:
                full_text_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"
            
            return EnhancedPubMedArticle(
                pmid=base.pmid,
                title=base.title,
                abstract=base.abstract,
                authors=base.authors,
                journal=base.journal,
                year=base.year,
                doi=base.doi,
                relevance_score=base.relevance_score,
                mesh_terms=mesh_terms,
                keywords=keywords,
                publication_types=pub_types,
                pmc_id=pmc_id,
                full_text_url=full_text_url,
                affiliations=affiliations[:5]
            )
            
        except Exception as e:
            logger.error(f"Error parsing enhanced article: {e}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def bulk_search(
        self,
        query: str,
        max_results: int = 500,
        batch_size: int = 100,
        year_range: Optional[Tuple[int, int]] = None,
        article_types: Optional[List[str]] = None
    ) -> List[EnhancedPubMedArticle]:
        """
        Bulk search with pagination for large result sets.
        
        Args:
            query: Search query
            max_results: Maximum total results
            batch_size: Results per batch
            year_range: (start_year, end_year) filter
            article_types: Filter by publication types
        """
        # Build query with filters
        full_query = query
        
        if year_range:
            full_query += f" AND {year_range[0]}:{year_range[1]}[pdat]"
        
        if article_types:
            type_filter = " OR ".join([f"{t}[pt]" for t in article_types])
            full_query += f" AND ({type_filter})"
        
        all_articles = []
        
        for start in range(0, max_results, batch_size):
            remaining = min(batch_size, max_results - start)
            
            # Search
            params = {
                'db': 'pubmed',
                'term': full_query,
                'retmax': remaining,
                'retstart': start,
                'sort': 'relevance',
                'retmode': 'xml'
            }
            
            try:
                response = self._make_request('esearch.fcgi', params)
                root = ET.fromstring(response.content)
                
                pmids = [elem.text for elem in root.findall('.//Id') if elem.text]
                
                if not pmids:
                    break
                
                # Fetch details
                articles = self.fetch_enhanced_details(pmids)
                all_articles.extend(articles)
                
                logger.info(f"Bulk search: fetched {len(all_articles)}/{max_results} articles")
                
                if len(pmids) < remaining:
                    break  # No more results
                    
            except Exception as e:
                logger.error(f"Bulk search error at offset {start}: {e}")
                break
        
        # Add relevance scores
        for i, article in enumerate(all_articles):
            article.relevance_score = 1.0 - (i / max(len(all_articles), 1))
        
        return all_articles
    
    def fetch_enhanced_details(self, pmids: List[str]) -> List[EnhancedPubMedArticle]:
        """Fetch enhanced article details."""
        if not pmids:
            return []
        
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'rettype': 'abstract'
        }
        
        try:
            response = self._make_request('efetch.fcgi', params)
            root = ET.fromstring(response.content)
            
            articles = []
            for article_elem in root.findall('.//PubmedArticle'):
                article = self._parse_enhanced_article(article_elem)
                if article:
                    articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching enhanced details: {e}")
            return []
    
    def score_articles(
        self,
        articles: List[EnhancedPubMedArticle],
        target_conditions: List[str],
        boost_recent: bool = True
    ) -> List[EnhancedPubMedArticle]:
        """
        Score articles based on medical ontology relevance.
        
        Args:
            articles: List of articles to score
            target_conditions: Conditions to match (e.g., ["DIABETES", "CARDIOVASCULAR"])
            boost_recent: Boost score for recent publications
        """
        current_year = datetime.now().year
        
        for article in articles:
            score = 0.0
            
            for condition in target_conditions:
                ontology = MEDICAL_ONTOLOGY.get(condition.upper(), {})
                condition_score = 0.0
                
                # Score based on MeSH terms
                for mesh in article.mesh_terms:
                    mesh_lower = mesh.lower()
                    for ont_mesh in ontology.get("mesh_terms", []):
                        if ont_mesh.lower() in mesh_lower or mesh_lower in ont_mesh.lower():
                            condition_score += 0.3
                
                # Score based on keywords
                article_text = f"{article.title} {article.abstract}".lower()
                for keyword in ontology.get("keywords", []):
                    if keyword.lower() in article_text:
                        condition_score += 0.1
                
                score += min(condition_score, 1.0) * ontology.get("weight", 1.0)
            
            # Normalize by number of conditions
            if target_conditions:
                score /= len(target_conditions)
            
            # Boost for recency
            if boost_recent and article.year:
                years_old = current_year - article.year
                recency_boost = max(0, 0.2 - (years_old * 0.02))
                score += recency_boost
            
            # Boost for clinical trials
            if any("Clinical Trial" in pt for pt in article.publication_types):
                score += 0.15
            
            # Boost for full text availability
            if article.pmc_id:
                score += 0.1
            
            article.relevance_score = min(score, 1.0)
        
        # Sort by score
        articles.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return articles
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_cited_by(self, pmid: str, max_results: int = 50) -> List[str]:
        """Get PMIDs of articles that cite this article."""
        params = {
            'dbfrom': 'pubmed',
            'db': 'pubmed',
            'id': pmid,
            'linkname': 'pubmed_pubmed_citedin',
            'retmode': 'xml'
        }
        
        try:
            response = self._make_request('elink.fcgi', params)
            root = ET.fromstring(response.content)
            
            citing_pmids = []
            for link_elem in root.findall('.//Link/Id'):
                if link_elem.text:
                    citing_pmids.append(link_elem.text)
            
            return citing_pmids[:max_results]
            
        except Exception as e:
            logger.error(f"Error getting citations for {pmid}: {e}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_references(self, pmid: str, max_results: int = 50) -> List[str]:
        """Get PMIDs of articles referenced by this article."""
        params = {
            'dbfrom': 'pubmed',
            'db': 'pubmed',
            'id': pmid,
            'linkname': 'pubmed_pubmed_refs',
            'retmode': 'xml'
        }
        
        try:
            response = self._make_request('elink.fcgi', params)
            root = ET.fromstring(response.content)
            
            ref_pmids = []
            for link_elem in root.findall('.//Link/Id'):
                if link_elem.text:
                    ref_pmids.append(link_elem.text)
            
            return ref_pmids[:max_results]
            
        except Exception as e:
            logger.error(f"Error getting references for {pmid}: {e}")
            return []
    
    def get_citation_network(
        self,
        seed_pmid: str,
        depth: int = 1,
        max_per_level: int = 20
    ) -> Dict[str, Any]:
        """
        Build citation network from seed article.
        
        Returns graph with articles and citation relationships.
        """
        visited: Set[str] = set()
        articles: Dict[str, EnhancedPubMedArticle] = {}
        edges: List[Tuple[str, str]] = []  # (citing, cited)
        
        def explore(pmid: str, current_depth: int):
            if pmid in visited or current_depth > depth:
                return
            
            visited.add(pmid)
            
            # Fetch article
            fetched = self.fetch_enhanced_details([pmid])
            if fetched:
                articles[pmid] = fetched[0]
            
            if current_depth < depth:
                # Get citing articles
                citing = self.get_cited_by(pmid, max_per_level)
                for citing_pmid in citing:
                    edges.append((citing_pmid, pmid))
                    explore(citing_pmid, current_depth + 1)
                
                # Get referenced articles
                refs = self.get_references(pmid, max_per_level)
                for ref_pmid in refs:
                    edges.append((pmid, ref_pmid))
        
        explore(seed_pmid, 0)
        
        return {
            "seed_pmid": seed_pmid,
            "articles": {pmid: a.to_dict() for pmid, a in articles.items()},
            "edges": edges,
            "total_articles": len(articles),
            "total_edges": len(edges)
        }
    
    def search_with_mesh_expansion(
        self,
        conditions: List[str],
        max_results: int = 100
    ) -> List[EnhancedPubMedArticle]:
        """Search using expanded MeSH terms from ontology."""
        mesh_terms = []
        
        for condition in conditions:
            ontology = MEDICAL_ONTOLOGY.get(condition.upper(), {})
            mesh_terms.extend(ontology.get("mesh_terms", []))
        
        if not mesh_terms:
            return []
        
        # Build MeSH query
        mesh_query = " OR ".join([f'"{term}"[MeSH Terms]' for term in mesh_terms[:10]])
        query = f"({mesh_query})"
        
        return self.bulk_search(query, max_results)
