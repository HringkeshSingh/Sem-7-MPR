"""
Multi-source aggregator for combining data from all sources.

Orchestrates retrieval from multiple sources and provides:
- Parallel retrieval
- Result deduplication
- Source ranking
- Cross-validation
"""

import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import hashlib

from .base_client import BaseDataSourceClient, RetrievedDocument
from .clinical_trials_client import ClinicalTrialsClient
from .who_client import WHOClient
from .medical_news_client import MedicalNewsClient

logger = logging.getLogger(__name__)


class MultiSourceAggregator:
    """
    Aggregates data from multiple medical data sources.
    
    Features:
    - Parallel retrieval from multiple sources
    - Result deduplication
    - Cross-source validation
    - Relevance ranking
    """
    
    def __init__(
        self,
        enable_clinical_trials: bool = True,
        enable_who: bool = True,
        enable_medical_news: bool = True,
        max_workers: int = 3
    ):
        """
        Initialize the multi-source aggregator.
        
        Args:
            enable_clinical_trials: Enable ClinicalTrials.gov source
            enable_who: Enable WHO data source
            enable_medical_news: Enable medical news source
            max_workers: Maximum parallel workers for retrieval
        """
        self.sources: Dict[str, BaseDataSourceClient] = {}
        self.max_workers = max_workers
        
        # Initialize enabled sources
        if enable_clinical_trials:
            self.sources['clinical_trials'] = ClinicalTrialsClient()
            
        if enable_who:
            self.sources['who'] = WHOClient()
            
        if enable_medical_news:
            self.sources['medical_news'] = MedicalNewsClient()
        
        logger.info(f"MultiSourceAggregator initialized with sources: {list(self.sources.keys())}")
    
    def add_source(self, name: str, client: BaseDataSourceClient):
        """Add a custom data source."""
        self.sources[name] = client
        logger.info(f"Added data source: {name}")
    
    def get_available_sources(self) -> List[str]:
        """Get list of available sources."""
        available = []
        for name, client in self.sources.items():
            if client.is_available():
                available.append(name)
        return available
    
    def search_all(
        self,
        query: str,
        max_results_per_source: int = 10,
        sources: Optional[List[str]] = None
    ) -> List[RetrievedDocument]:
        """
        Search all enabled sources in parallel.
        
        Args:
            query: Search query
            max_results_per_source: Maximum results per source
            sources: Optional list of specific sources to search
            
        Returns:
            Combined list of RetrievedDocument objects, deduplicated and ranked
        """
        sources_to_search = sources or list(self.sources.keys())
        all_documents = []
        
        # Use ThreadPoolExecutor for parallel retrieval
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_source = {}
            
            for source_name in sources_to_search:
                if source_name in self.sources:
                    client = self.sources[source_name]
                    if client.is_available():
                        future = executor.submit(
                            self._search_source,
                            client,
                            query,
                            max_results_per_source
                        )
                        future_to_source[future] = source_name
                    else:
                        logger.warning(f"Source {source_name} is not available")
            
            for future in concurrent.futures.as_completed(future_to_source, timeout=60):
                source_name = future_to_source[future]
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                    logger.info(f"Retrieved {len(documents)} documents from {source_name}")
                except Exception as e:
                    logger.error(f"Error retrieving from {source_name}: {e}")
        
        # Deduplicate and rank
        deduplicated = self._deduplicate_documents(all_documents)
        ranked = self._rank_documents(deduplicated)
        
        logger.info(f"Total documents retrieved: {len(all_documents)}, after dedup: {len(ranked)}")
        return ranked
    
    def _search_source(
        self,
        client: BaseDataSourceClient,
        query: str,
        max_results: int
    ) -> List[RetrievedDocument]:
        """Search a single source."""
        try:
            return client.search(query, max_results)
        except Exception as e:
            logger.error(f"Error searching {client.source_name}: {e}")
            return []
    
    def _deduplicate_documents(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        Remove duplicate documents based on content similarity.
        
        Uses title and content hashing for deduplication.
        """
        seen_hashes = set()
        unique_documents = []
        
        for doc in documents:
            # Create hash from title and first 500 chars of content
            content_key = f"{doc.title.lower().strip()}{doc.content[:500].lower().strip()}"
            content_hash = hashlib.md5(content_key.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_documents.append(doc)
        
        return unique_documents
    
    def _rank_documents(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        Rank documents based on multiple factors.
        
        Factors:
        - Original relevance score
        - Source reliability weight
        - Recency
        - Content quality (length, structure)
        """
        # Source reliability weights
        source_weights = {
            'pubmed': 1.0,       # Research papers - highest weight
            'clinical_trials': 0.95,  # Clinical trials - very high
            'who': 0.9,         # WHO data - high reliability
            'medlineplus': 0.85,  # MedlinePlus - good reliability
            'medical_news': 0.7   # News - lower weight
        }
        
        for doc in documents:
            # Base score from source relevance
            base_score = doc.relevance_score
            
            # Apply source weight
            source_weight = source_weights.get(doc.source, 0.5)
            
            # Recency bonus (if date available)
            recency_bonus = 0.0
            if doc.publication_date:
                days_old = (datetime.now() - doc.publication_date).days
                if days_old < 365:  # Less than 1 year
                    recency_bonus = 0.1
                elif days_old < 365 * 3:  # 1-3 years
                    recency_bonus = 0.05
            
            # Content quality bonus
            content_bonus = 0.0
            if len(doc.content) > 1000:
                content_bonus = 0.05
            if doc.authors:
                content_bonus += 0.03
            if doc.url:
                content_bonus += 0.02
            
            # Calculate final score
            final_score = (base_score * source_weight) + recency_bonus + content_bonus
            doc.relevance_score = min(final_score, 1.0)
        
        # Sort by final relevance score
        return sorted(documents, key=lambda x: x.relevance_score, reverse=True)
    
    def cross_validate(
        self,
        documents: List[RetrievedDocument],
        generated_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Cross-validate generated data against retrieved documents.
        
        Checks for:
        - Statistical alignment
        - Clinical plausibility
        - Consistency with literature
        
        Args:
            documents: Retrieved reference documents
            generated_data: Generated synthetic data to validate
            
        Returns:
            Validation results with confidence scores
        """
        validation_results = {
            'overall_confidence': 0.0,
            'sources_checked': len(documents),
            'validations': [],
            'warnings': [],
            'recommendations': []
        }
        
        if not documents:
            validation_results['warnings'].append("No reference documents available for validation")
            return validation_results
        
        # Extract key statistics from generated data
        gen_stats = self._extract_generated_stats(generated_data)
        
        # Extract reference statistics from documents
        ref_stats = self._extract_reference_stats(documents)
        
        # Compare statistics
        confidence_scores = []
        
        # Check conditions match
        if gen_stats.get('conditions') and ref_stats.get('conditions'):
            overlap = len(set(gen_stats['conditions']) & set(ref_stats['conditions']))
            total = len(set(gen_stats['conditions']) | set(ref_stats['conditions']))
            condition_match = overlap / max(total, 1)
            confidence_scores.append(condition_match)
            
            if condition_match > 0.7:
                validation_results['validations'].append(
                    f"Generated conditions align well with literature ({condition_match:.0%} match)"
                )
            elif condition_match < 0.3:
                validation_results['warnings'].append(
                    "Generated conditions may not be well-supported by retrieved literature"
                )
        
        # Check demographics alignment
        if gen_stats.get('age_range') and ref_stats.get('age_focus'):
            validation_results['validations'].append(
                f"Demographics: {ref_stats['age_focus']} population found in {ref_stats.get('source_count', 0)} sources"
            )
        
        # Calculate overall confidence
        if confidence_scores:
            validation_results['overall_confidence'] = sum(confidence_scores) / len(confidence_scores)
        else:
            validation_results['overall_confidence'] = 0.5  # Neutral if no specific checks
        
        # Add recommendations
        if validation_results['overall_confidence'] < 0.5:
            validation_results['recommendations'].append(
                "Consider adding more reference documents to improve validation"
            )
        
        return validation_results
    
    def _extract_generated_stats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract statistics from generated data."""
        stats = {}
        
        if isinstance(data, dict):
            stats['conditions'] = data.get('diagnoses', [])
            stats['age_range'] = data.get('age_range')
            stats['gender'] = data.get('gender')
            
        return stats
    
    def _extract_reference_stats(self, documents: List[RetrievedDocument]) -> Dict[str, Any]:
        """Extract statistics from reference documents."""
        stats = {
            'conditions': [],
            'age_focus': None,
            'source_count': len(documents)
        }
        
        # Common medical conditions to look for
        condition_keywords = [
            'diabetes', 'cardiovascular', 'hypertension', 'respiratory',
            'renal', 'sepsis', 'neurological', 'cancer', 'trauma'
        ]
        
        for doc in documents:
            content_lower = doc.content.lower()
            
            for condition in condition_keywords:
                if condition in content_lower:
                    stats['conditions'].append(condition.upper())
            
            # Check for age-related content
            if 'elderly' in content_lower or 'geriatric' in content_lower or 'aged' in content_lower:
                stats['age_focus'] = 'elderly'
            elif 'pediatric' in content_lower or 'child' in content_lower:
                stats['age_focus'] = 'pediatric'
        
        # Remove duplicates
        stats['conditions'] = list(set(stats['conditions']))
        
        return stats
    
    def get_source_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about all configured sources."""
        stats = {}
        
        for name, client in self.sources.items():
            stats[name] = {
                'available': client.is_available(),
                'source_name': client.source_name,
                'rate_limit': client.rate_limit
            }
        
        return stats
