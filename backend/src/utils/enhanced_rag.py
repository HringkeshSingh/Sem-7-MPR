"""
Enhanced RAG System with Multi-Source Retrieval and Cross-Validation.

This module provides an advanced RAG system that:
1. Retrieves data from multiple sources (PubMed, ClinicalTrials.gov, WHO, etc.)
2. Stores and indexes documents in ChromaDB for efficient semantic search
3. Cross-validates generated data against retrieved literature
4. Provides confidence-scored responses with source attribution
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

from langchain_core.documents import Document

from config.settings import MODELS_DIR
from src.utils.logging_config import get_logger
from src.utils.rag_system import HealthcareRAGSystem
from src.utils.data_sources import MultiSourceAggregator, RetrievedDocument
from src.utils.cross_validator import CrossValidator, CrossValidationReport
from src.utils.pubmed_client import PubMedClient

logger = get_logger(__name__)


class EnhancedRAGSystem:
    """
    Advanced RAG system with multi-source retrieval and validation.
    
    Features:
    - Multi-source data retrieval (PubMed, ClinicalTrials.gov, WHO, MedlinePlus)
    - ChromaDB vector storage with semantic search
    - Cross-validation of generated data against literature
    - Source reliability scoring
    - Comprehensive confidence metrics
    
    Usage:
        >>> rag = EnhancedRAGSystem()
        >>> result = rag.retrieve_and_extract("diabetes in elderly patients")
        >>> print(result['summary'])
        >>> validation = rag.validate_generated_data(synthetic_data)
    """
    
    def __init__(
        self,
        vector_store_path: Optional[Path] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_clinical_trials: bool = True,
        enable_who: bool = True,
        enable_medical_news: bool = True,
        enable_pubmed: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 10
    ):
        """
        Initialize the enhanced RAG system.
        
        Args:
            vector_store_path: Path for vector database storage
            embedding_model: Sentence transformer model for embeddings
            enable_clinical_trials: Enable ClinicalTrials.gov source
            enable_who: Enable WHO data source
            enable_medical_news: Enable medical news source
            enable_pubmed: Enable PubMed source
            chunk_size: Text chunk size for splitting
            chunk_overlap: Overlap between chunks
            top_k: Number of top results to retrieve
        """
        # Initialize base RAG system
        self._base_rag = HealthcareRAGSystem(
            vector_store_path=vector_store_path,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k
        )
        
        # Initialize multi-source aggregator
        self._aggregator = MultiSourceAggregator(
            enable_clinical_trials=enable_clinical_trials,
            enable_who=enable_who,
            enable_medical_news=enable_medical_news
        )
        
        # Initialize PubMed client separately for more control
        self._pubmed = PubMedClient() if enable_pubmed else None
        
        # Initialize cross-validator
        self._validator = CrossValidator()
        
        # Cache for retrieved documents
        self._document_cache: Dict[str, List[RetrievedDocument]] = {}
        self._cache_ttl = 3600  # 1 hour cache
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Statistics
        self._stats = {
            'total_queries': 0,
            'documents_retrieved': 0,
            'documents_indexed': 0,
            'validations_performed': 0
        }
        
        logger.info("EnhancedRAGSystem initialized successfully")
    
    def retrieve_and_extract(
        self,
        query: str,
        max_results_per_source: int = 10,
        use_cache: bool = True,
        index_results: bool = True,
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve from multiple sources and extract relevant information.
        
        Args:
            query: Search query
            max_results_per_source: Maximum results from each source
            use_cache: Use cached results if available
            index_results: Add retrieved documents to vector store
            sources: Specific sources to query (None for all)
            
        Returns:
            Dictionary with extracted information, sources, and confidence
        """
        self._stats['total_queries'] += 1
        start_time = datetime.now()
        
        # Check cache
        cache_key = f"{query}:{max_results_per_source}:{str(sources)}"
        if use_cache and cache_key in self._document_cache:
            cache_time = self._cache_timestamps.get(cache_key)
            if cache_time and (datetime.now() - cache_time).seconds < self._cache_ttl:
                logger.info(f"Using cached results for query: {query[:50]}...")
                documents = self._document_cache[cache_key]
            else:
                documents = self._fetch_from_sources(query, max_results_per_source, sources)
        else:
            documents = self._fetch_from_sources(query, max_results_per_source, sources)
        
        # Cache results
        self._document_cache[cache_key] = documents
        self._cache_timestamps[cache_key] = datetime.now()
        
        # Index in vector store if requested
        if index_results and documents:
            self._index_documents(documents)
        
        # Extract information using base RAG
        rag_result = self._base_rag.extract_relevant_info(query)
        
        # Enhance with source information
        source_summary = self._summarize_sources(documents)
        
        # Calculate enhanced confidence
        confidence = self._calculate_enhanced_confidence(rag_result, documents)
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'query': query,
            'relevant_info': rag_result.get('relevant_info', []),
            'summary': rag_result.get('summary', ''),
            'confidence': confidence,
            'sources': source_summary,
            'source_documents': [doc.to_dict() for doc in documents[:20]],  # Increased from 10
            'num_documents_retrieved': len(documents),
            'num_sources_queried': len(source_summary.get('by_source', {})),
            'processing_time': elapsed_time,
            'from_cache': use_cache and cache_key in self._document_cache
        }
    
    def _fetch_from_sources(
        self,
        query: str,
        max_results: int,
        sources: Optional[List[str]]
    ) -> List[RetrievedDocument]:
        """Fetch documents from all configured sources."""
        all_documents = []
        
        # Fetch from multi-source aggregator
        try:
            agg_docs = self._aggregator.search_all(
                query,
                max_results_per_source=max_results,
                sources=sources
            )
            all_documents.extend(agg_docs)
        except Exception as e:
            logger.error(f"Error fetching from aggregator: {e}")
        
        # Fetch from PubMed
        if self._pubmed:
            try:
                pubmed_articles = self._pubmed.search_and_fetch(query, max_results)
                for article in pubmed_articles:
                    doc = RetrievedDocument(
                        source="pubmed",
                        source_id=article.pmid,
                        title=article.title,
                        content=f"{article.title}\n\n{article.abstract}",
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/",
                        authors=article.authors,
                        publication_date=datetime(article.year, 1, 1) if article.year else None,
                        metadata={
                            'journal': article.journal,
                            'doi': article.doi,
                            'pmid': article.pmid
                        },
                        relevance_score=article.relevance_score
                    )
                    all_documents.append(doc)
            except Exception as e:
                logger.error(f"Error fetching from PubMed: {e}")
        
        self._stats['documents_retrieved'] += len(all_documents)
        logger.info(f"Retrieved {len(all_documents)} total documents for query: {query[:50]}...")
        
        return all_documents
    
    def _index_documents(self, documents: List[RetrievedDocument]):
        """Add retrieved documents to the vector store."""
        try:
            texts = []
            metadatas = []
            
            for doc in documents:
                texts.append(doc.to_text())
                metadatas.append({
                    'source': doc.source,
                    'source_id': doc.source_id,
                    'title': doc.title,
                    'url': doc.url or '',
                    'relevance_score': doc.relevance_score
                })
            
            if texts:
                count = self._base_rag.add_texts(texts, metadatas)
                self._stats['documents_indexed'] += count
                logger.info(f"Indexed {count} document chunks to vector store")
                
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
    
    def _summarize_sources(self, documents: List[RetrievedDocument]) -> Dict[str, Any]:
        """Create a summary of source information."""
        by_source = {}
        
        for doc in documents:
            if doc.source not in by_source:
                by_source[doc.source] = {
                    'count': 0,
                    'avg_relevance': 0.0,
                    'titles': []
                }
            
            by_source[doc.source]['count'] += 1
            by_source[doc.source]['avg_relevance'] += doc.relevance_score
            by_source[doc.source]['titles'].append(doc.title[:100])
        
        # Calculate averages
        for source in by_source:
            count = by_source[source]['count']
            by_source[source]['avg_relevance'] /= max(count, 1)
            by_source[source]['titles'] = by_source[source]['titles'][:5]  # Limit
        
        return {
            'total_documents': len(documents),
            'unique_sources': len(by_source),
            'by_source': by_source
        }
    
    def _calculate_enhanced_confidence(
        self,
        rag_result: Dict[str, Any],
        documents: List[RetrievedDocument]
    ) -> float:
        """Calculate enhanced confidence score."""
        base_confidence = rag_result.get('confidence', 0.0)
        
        # Source diversity bonus
        sources = set(doc.source for doc in documents)
        diversity_bonus = min(len(sources) * 0.05, 0.15)
        
        # Document count factor
        doc_factor = min(len(documents) / 20, 0.1)
        
        # Average source relevance
        avg_relevance = 0.0
        if documents:
            avg_relevance = sum(doc.relevance_score for doc in documents) / len(documents)
        
        # Combined confidence
        combined = (
            base_confidence * 0.5 +
            avg_relevance * 0.3 +
            diversity_bonus +
            doc_factor
        )
        
        return min(max(combined, 0.0), 1.0)
    
    def validate_generated_data(
        self,
        generated_data: Dict[str, Any],
        query: Optional[str] = None,
        use_existing_documents: bool = True
    ) -> CrossValidationReport:
        """
        Cross-validate generated data against retrieved literature.
        
        Args:
            generated_data: Generated synthetic data to validate
            query: Original query (used to retrieve more documents if needed)
            use_existing_documents: Use documents from previous queries
            
        Returns:
            CrossValidationReport with validation results
        """
        self._stats['validations_performed'] += 1
        
        # Collect reference documents
        reference_docs = []
        
        # Use cached documents
        if use_existing_documents:
            for docs in self._document_cache.values():
                reference_docs.extend(docs)
        
        # Fetch more if we have a query and need more documents
        if query and len(reference_docs) < 10:
            new_docs = self._fetch_from_sources(query, 15, None)
            reference_docs.extend(new_docs)
        
        # Deduplicate
        seen_ids = set()
        unique_docs = []
        for doc in reference_docs:
            doc_key = f"{doc.source}:{doc.source_id}"
            if doc_key not in seen_ids:
                seen_ids.add(doc_key)
                unique_docs.append(doc)
        
        # Extract query context for validation
        query_context = None
        if query:
            rag_result = self._base_rag.extract_relevant_info(query)
            filtered = self._base_rag.filter_query_parameters(query, rag_result)
            query_context = filtered
        
        # Perform validation
        report = self._validator.validate(
            generated_data=generated_data,
            reference_documents=unique_docs,
            query_context=query_context
        )
        
        logger.info(
            f"Cross-validation complete: {report.checks_passed}/{report.total_checks} passed, "
            f"score: {report.overall_score:.2f}"
        )
        
        return report
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources."""
        sources = self._aggregator.get_available_sources()
        if self._pubmed:
            sources.append('pubmed')
        return sources
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        base_stats = self._base_rag.get_vectorstore_stats()
        source_stats = self._aggregator.get_source_stats()
        
        return {
            **self._stats,
            'vectorstore': base_stats,
            'sources': source_stats,
            'cache_size': len(self._document_cache),
            'available_sources': self.get_available_sources()
        }
    
    def clear_cache(self):
        """Clear the document cache."""
        self._document_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Document cache cleared")
    
    def add_custom_documents(
        self,
        documents: List[Dict[str, Any]],
        source_name: str = "custom"
    ) -> int:
        """
        Add custom documents to the knowledge base.
        
        Args:
            documents: List of document dictionaries with 'title' and 'content'
            source_name: Name to identify the source
            
        Returns:
            Number of documents indexed
        """
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            texts.append(f"{doc.get('title', '')}\n\n{doc.get('content', '')}")
            metadatas.append({
                'source': source_name,
                'source_id': f"{source_name}_{i}",
                'title': doc.get('title', f'Document {i}'),
                'url': doc.get('url', '')
            })
        
        if texts:
            count = self._base_rag.add_texts(texts, metadatas)
            logger.info(f"Added {count} custom document chunks")
            return count
        
        return 0
    
    # Delegate to base RAG for basic operations
    def extract_relevant_info(self, query: str, max_docs: Optional[int] = None) -> Dict[str, Any]:
        """Extract information from existing knowledge base."""
        return self._base_rag.extract_relevant_info(query, max_docs)
    
    def filter_query_parameters(self, query: str, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Filter and extract query parameters."""
        return self._base_rag.filter_query_parameters(query, extracted_info)
