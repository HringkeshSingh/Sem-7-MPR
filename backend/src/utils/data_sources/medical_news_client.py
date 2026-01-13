"""
Medical news and guidelines client.

Retrieves medical news and guidelines from various sources.
This is a placeholder for medical news APIs - can be extended with:
- MedlinePlus
- Medical news aggregators
- FDA alerts
"""

import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .base_client import BaseDataSourceClient, RetrievedDocument

logger = logging.getLogger(__name__)


class MedicalNewsClient(BaseDataSourceClient):
    """
    Client for retrieving medical news and guidelines.
    
    Sources:
    - MedlinePlus Connect API
    - FDA Drug information
    """
    
    MEDLINEPLUS_URL = "https://connect.medlineplus.gov/service"
    
    def __init__(self, rate_limit: float = 1.0):
        """Initialize the medical news client."""
        super().__init__(source_name="medical_news", rate_limit=rate_limit)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HealthcareDataGenerator/1.0'
        })
    
    def _check_availability(self) -> bool:
        """Check if MedlinePlus API is accessible."""
        try:
            # MedlinePlus doesn't have a simple health endpoint
            # Just check if we can connect
            response = self.session.head(
                "https://medlineplus.gov",
                timeout=10
            )
            return response.status_code in [200, 301, 302]
        except Exception as e:
            logger.warning(f"MedlinePlus not available: {e}")
            return False
    
    def search(self, query: str, max_results: int = 20) -> List[RetrievedDocument]:
        """
        Search for medical information.
        
        Args:
            query: Search query (conditions, medications, etc.)
            max_results: Maximum number of results
            
        Returns:
            List of RetrievedDocument objects
        """
        documents = []
        
        # Try to get MedlinePlus health topics
        try:
            medlineplus_docs = self._search_medlineplus(query, max_results)
            documents.extend(medlineplus_docs)
        except Exception as e:
            logger.warning(f"Error searching MedlinePlus: {e}")
        
        # Try to get drug information if query mentions medications
        try:
            drug_docs = self._search_drug_info(query, max_results // 2)
            documents.extend(drug_docs)
        except Exception as e:
            logger.warning(f"Error searching drug info: {e}")
        
        logger.info(f"Retrieved {len(documents)} medical documents for query: {query}")
        return documents[:max_results]
    
    def _search_medlineplus(self, query: str, max_results: int) -> List[RetrievedDocument]:
        """Search MedlinePlus health topics."""
        self._rate_limit_wait()
        
        documents = []
        
        # Map query to ICD codes for MedlinePlus Connect
        icd_codes = self._map_to_icd_codes(query)
        
        for code in icd_codes[:max_results]:
            try:
                params = {
                    "mainSearchCriteria.v.cs": "2.16.840.1.113883.6.90",
                    "mainSearchCriteria.v.c": code,
                    "knowledgeResponseType": "application/json"
                }
                
                response = self.session.get(
                    self.MEDLINEPLUS_URL,
                    params=params,
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    doc = self._parse_medlineplus_response(data, code)
                    if doc:
                        documents.append(doc)
                        
            except Exception as e:
                logger.debug(f"Error getting MedlinePlus data for {code}: {e}")
                continue
        
        return documents
    
    def _map_to_icd_codes(self, query: str) -> List[str]:
        """Map query terms to ICD-10 codes."""
        icd_mapping = {
            'diabetes': ['E11', 'E11.9'],
            'hypertension': ['I10', 'I11'],
            'cardiovascular': ['I25', 'I50'],
            'heart': ['I25', 'I51'],
            'respiratory': ['J44', 'J45'],
            'copd': ['J44'],
            'asthma': ['J45'],
            'renal': ['N18', 'N19'],
            'kidney': ['N18', 'N19'],
            'sepsis': ['A41'],
            'cancer': ['C80'],
            'stroke': ['I63', 'I64'],
            'neurological': ['G00', 'G89']
        }
        
        codes = []
        query_lower = query.lower()
        
        for term, term_codes in icd_mapping.items():
            if term in query_lower:
                codes.extend(term_codes)
        
        return codes[:10] if codes else ['E11']  # Default to diabetes if no match
    
    def _parse_medlineplus_response(self, data: Dict, code: str) -> Optional[RetrievedDocument]:
        """Parse MedlinePlus API response."""
        try:
            feed = data.get("feed", {})
            entries = feed.get("entry", [])
            
            if not entries:
                return None
            
            entry = entries[0]
            title = entry.get("title", {}).get("_value", "Health Information")
            
            # Get summary
            summary_list = entry.get("summary", {}).get("_value", "")
            if isinstance(summary_list, list):
                summary = " ".join(summary_list)
            else:
                summary = summary_list
            
            # Get link
            links = entry.get("link", [])
            url = None
            if links:
                url = links[0].get("href")
            
            content = f"{title}\n\n{summary}"
            
            return RetrievedDocument(
                source="medlineplus",
                source_id=f"medlineplus_{code}",
                title=title,
                content=content[:3000],
                url=url,
                metadata={
                    "icd_code": code,
                    "source_type": "health_topic"
                },
                relevance_score=0.9
            )
            
        except Exception as e:
            logger.warning(f"Error parsing MedlinePlus response: {e}")
            return None
    
    def _search_drug_info(self, query: str, max_results: int) -> List[RetrievedDocument]:
        """Search for drug/medication information."""
        # This could be extended to use openFDA or DailyMed APIs
        # For now, return empty list
        return []
    
    def fetch_by_id(self, doc_id: str) -> Optional[RetrievedDocument]:
        """Fetch a specific document by ID."""
        # Parse the doc_id to determine source
        if doc_id.startswith("medlineplus_"):
            code = doc_id.replace("medlineplus_", "")
            docs = self._search_medlineplus(code, 1)
            return docs[0] if docs else None
        return None
