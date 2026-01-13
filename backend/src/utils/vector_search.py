"""
Vector Search Interface.

Provides advanced search capabilities:
- Semantic search with filters
- Hybrid search (keyword + semantic)
- Relevance ranking and reranking
- Multi-collection search
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

from src.utils.vector_db_manager import VectorDBManager, Document, QueryResult
from src.utils.embedding_pipeline import EmbeddingPipeline

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Search mode options."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class RankingStrategy(Enum):
    """Ranking strategy options."""
    SEMANTIC_SCORE = "semantic"
    BM25 = "bm25"
    RECIPROCAL_RANK_FUSION = "rrf"
    WEIGHTED_COMBINATION = "weighted"


@dataclass
class SearchQuery:
    """Search query with parameters."""
    text: str
    filters: Optional[Dict[str, Any]] = None
    collections: List[str] = field(default_factory=lambda: ["default"])
    top_k: int = 10
    mode: SearchMode = SearchMode.SEMANTIC
    rerank: bool = False
    min_score: float = 0.0


@dataclass
class SearchResult:
    """Search result with ranking information."""
    document_id: str
    content: str
    score: float
    rank: int
    collection: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    highlights: List[str] = field(default_factory=list)
    explanation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "content": self.content,
            "score": self.score,
            "rank": self.rank,
            "collection": self.collection,
            "metadata": self.metadata,
            "highlights": self.highlights
        }


class KeywordSearcher:
    """Simple keyword-based search using BM25-like scoring."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._documents: Dict[str, Document] = {}
        self._inverted_index: Dict[str, Set[str]] = defaultdict(set)
        self._doc_lengths: Dict[str, int] = {}
        self._avg_doc_length: float = 0.0
    
    def index_documents(self, documents: List[Document]):
        """Build inverted index from documents."""
        total_length = 0
        
        for doc in documents:
            self._documents[doc.id] = doc
            tokens = self._tokenize(doc.content)
            self._doc_lengths[doc.id] = len(tokens)
            total_length += len(tokens)
            
            for token in set(tokens):
                self._inverted_index[token].add(doc.id)
        
        if documents:
            self._avg_doc_length = total_length / len(documents)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_ids: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """Search documents using BM25."""
        query_tokens = self._tokenize(query)
        scores: Dict[str, float] = defaultdict(float)
        
        N = len(self._documents)
        
        for token in query_tokens:
            if token not in self._inverted_index:
                continue
            
            matching_docs = self._inverted_index[token]
            if doc_ids:
                matching_docs = matching_docs & doc_ids
            
            df = len(matching_docs)
            idf = max(0, (N - df + 0.5) / (df + 0.5))
            
            for doc_id in matching_docs:
                doc = self._documents[doc_id]
                doc_tokens = self._tokenize(doc.content)
                tf = doc_tokens.count(token)
                doc_len = self._doc_lengths.get(doc_id, 1)
                
                # BM25 scoring
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_doc_length, 1))
                score = idf * (numerator / denominator)
                
                scores[doc_id] += score
        
        # Sort by score
        results = sorted(scores.items(), key=lambda x: -x[1])
        return results[:top_k]


class VectorSearch:
    """
    Advanced vector search with hybrid capabilities.
    
    Usage:
        search = VectorSearch()
        
        # Simple semantic search
        results = search.semantic_search("diabetes treatment guidelines", top_k=10)
        
        # Hybrid search
        results = search.hybrid_search(
            "diabetes treatment",
            collections=["research", "clinical_trials"],
            alpha=0.7  # Weight for semantic vs keyword
        )
        
        # Search with filters
        results = search.search(SearchQuery(
            text="elderly diabetes",
            filters={"year": {"$gte": 2020}},
            mode=SearchMode.HYBRID,
            rerank=True
        ))
    """
    
    def __init__(
        self,
        db_manager: Optional[VectorDBManager] = None,
        embedding_pipeline: Optional[EmbeddingPipeline] = None
    ):
        self.db_manager = db_manager or VectorDBManager()
        self.embedding_pipeline = embedding_pipeline or EmbeddingPipeline()
        self.keyword_searcher = KeywordSearcher()
        self._indexed_collections: Set[str] = set()
    
    def index_for_keyword_search(self, collection: str = "default"):
        """Build keyword index for a collection."""
        if collection in self._indexed_collections:
            return
        
        # Get all documents from collection (for small collections)
        # For large collections, this should be done incrementally
        try:
            # This is a simplified approach - in production, you'd want pagination
            documents = self.db_manager.get([], collection)
            if documents:
                self.keyword_searcher.index_documents(documents)
                self._indexed_collections.add(collection)
                logger.info(f"Indexed {len(documents)} documents for keyword search")
        except Exception as e:
            logger.warning(f"Could not build keyword index: {e}")
    
    def semantic_search(
        self,
        query: str,
        collection: str = "default",
        top_k: int = 10,
        filters: Optional[Dict] = None,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """Pure semantic (vector) search."""
        # Generate query embedding
        query_embedding = self.embedding_pipeline.embed_text(query)
        
        # Search vector database
        results = self.db_manager.search(
            query_embedding=query_embedding,
            collection=collection,
            top_k=top_k,
            filters=filters
        )
        
        # Convert to SearchResult
        search_results = []
        for i, qr in enumerate(results):
            if qr.score >= min_score:
                search_results.append(SearchResult(
                    document_id=qr.document.id,
                    content=qr.document.content,
                    score=qr.score,
                    rank=i + 1,
                    collection=collection,
                    metadata=qr.document.metadata,
                    highlights=self._extract_highlights(qr.document.content, query)
                ))
        
        return search_results
    
    def keyword_search(
        self,
        query: str,
        collection: str = "default",
        top_k: int = 10
    ) -> List[SearchResult]:
        """Pure keyword (BM25) search."""
        self.index_for_keyword_search(collection)
        
        results = self.keyword_searcher.search(query, top_k)
        
        search_results = []
        for i, (doc_id, score) in enumerate(results):
            doc = self.keyword_searcher._documents.get(doc_id)
            if doc:
                search_results.append(SearchResult(
                    document_id=doc_id,
                    content=doc.content,
                    score=score,
                    rank=i + 1,
                    collection=collection,
                    metadata=doc.metadata,
                    highlights=self._extract_highlights(doc.content, query)
                ))
        
        return search_results
    
    def hybrid_search(
        self,
        query: str,
        collection: str = "default",
        top_k: int = 10,
        alpha: float = 0.7,
        filters: Optional[Dict] = None,
        ranking_strategy: RankingStrategy = RankingStrategy.RECIPROCAL_RANK_FUSION
    ) -> List[SearchResult]:
        """
        Hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query
            collection: Collection to search
            top_k: Number of results
            alpha: Weight for semantic search (0-1). 1=pure semantic, 0=pure keyword
            filters: Metadata filters
            ranking_strategy: How to combine results
        """
        # Get results from both methods
        semantic_results = self.semantic_search(query, collection, top_k * 2, filters)
        keyword_results = self.keyword_search(query, collection, top_k * 2)
        
        # Combine results based on strategy
        if ranking_strategy == RankingStrategy.RECIPROCAL_RANK_FUSION:
            combined = self._reciprocal_rank_fusion(semantic_results, keyword_results, top_k)
        elif ranking_strategy == RankingStrategy.WEIGHTED_COMBINATION:
            combined = self._weighted_combination(semantic_results, keyword_results, alpha, top_k)
        else:
            # Default to semantic
            combined = semantic_results[:top_k]
        
        return combined
    
    def _reciprocal_rank_fusion(
        self,
        results1: List[SearchResult],
        results2: List[SearchResult],
        top_k: int,
        k: int = 60
    ) -> List[SearchResult]:
        """Combine results using Reciprocal Rank Fusion."""
        scores: Dict[str, float] = defaultdict(float)
        docs: Dict[str, SearchResult] = {}
        
        for i, result in enumerate(results1):
            scores[result.document_id] += 1 / (k + i + 1)
            if result.document_id not in docs:
                docs[result.document_id] = result
        
        for i, result in enumerate(results2):
            scores[result.document_id] += 1 / (k + i + 1)
            if result.document_id not in docs:
                docs[result.document_id] = result
        
        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: -scores[x])
        
        combined = []
        for i, doc_id in enumerate(sorted_ids[:top_k]):
            result = docs[doc_id]
            result.score = scores[doc_id]
            result.rank = i + 1
            result.explanation = "RRF combined score"
            combined.append(result)
        
        return combined
    
    def _weighted_combination(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        alpha: float,
        top_k: int
    ) -> List[SearchResult]:
        """Combine results using weighted scores."""
        scores: Dict[str, float] = defaultdict(float)
        docs: Dict[str, SearchResult] = {}
        
        # Normalize and weight semantic scores
        if semantic_results:
            max_sem = max(r.score for r in semantic_results)
            for result in semantic_results:
                norm_score = result.score / max_sem if max_sem > 0 else 0
                scores[result.document_id] += alpha * norm_score
                if result.document_id not in docs:
                    docs[result.document_id] = result
        
        # Normalize and weight keyword scores
        if keyword_results:
            max_kw = max(r.score for r in keyword_results)
            for result in keyword_results:
                norm_score = result.score / max_kw if max_kw > 0 else 0
                scores[result.document_id] += (1 - alpha) * norm_score
                if result.document_id not in docs:
                    docs[result.document_id] = result
        
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: -scores[x])
        
        combined = []
        for i, doc_id in enumerate(sorted_ids[:top_k]):
            result = docs[doc_id]
            result.score = scores[doc_id]
            result.rank = i + 1
            result.explanation = f"Weighted: {alpha:.1f} semantic + {1-alpha:.1f} keyword"
            combined.append(result)
        
        return combined
    
    def multi_collection_search(
        self,
        query: str,
        collections: List[str],
        top_k: int = 10,
        mode: SearchMode = SearchMode.SEMANTIC,
        alpha: float = 0.7
    ) -> List[SearchResult]:
        """Search across multiple collections."""
        all_results = []
        
        for collection in collections:
            if mode == SearchMode.SEMANTIC:
                results = self.semantic_search(query, collection, top_k)
            elif mode == SearchMode.KEYWORD:
                results = self.keyword_search(query, collection, top_k)
            else:
                results = self.hybrid_search(query, collection, top_k, alpha)
            
            all_results.extend(results)
        
        # Re-rank across collections
        all_results.sort(key=lambda x: -x.score)
        
        # Update ranks
        for i, result in enumerate(all_results[:top_k]):
            result.rank = i + 1
        
        return all_results[:top_k]
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Main search method supporting all modes.
        
        Args:
            query: SearchQuery object with all parameters
        """
        if len(query.collections) > 1:
            results = self.multi_collection_search(
                query.text,
                query.collections,
                query.top_k,
                query.mode
            )
        else:
            collection = query.collections[0] if query.collections else "default"
            
            if query.mode == SearchMode.SEMANTIC:
                results = self.semantic_search(
                    query.text,
                    collection,
                    query.top_k,
                    query.filters,
                    query.min_score
                )
            elif query.mode == SearchMode.KEYWORD:
                results = self.keyword_search(query.text, collection, query.top_k)
            else:
                results = self.hybrid_search(
                    query.text,
                    collection,
                    query.top_k,
                    filters=query.filters
                )
        
        # Optional reranking
        if query.rerank:
            results = self._rerank_results(query.text, results)
        
        # Filter by minimum score
        results = [r for r in results if r.score >= query.min_score]
        
        return results
    
    def _rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = None
    ) -> List[SearchResult]:
        """Rerank results using cross-encoder or other method."""
        # Simple reranking based on query term overlap
        query_terms = set(query.lower().split())
        
        for result in results:
            content_terms = set(result.content.lower().split())
            overlap = len(query_terms & content_terms)
            # Boost score based on term overlap
            result.score = result.score * (1 + 0.1 * overlap)
        
        results.sort(key=lambda x: -x.score)
        
        for i, result in enumerate(results):
            result.rank = i + 1
        
        if top_k:
            return results[:top_k]
        return results
    
    def _extract_highlights(
        self,
        content: str,
        query: str,
        context_words: int = 10
    ) -> List[str]:
        """Extract relevant highlights from content."""
        highlights = []
        query_terms = query.lower().split()
        words = content.split()
        
        for i, word in enumerate(words):
            if word.lower() in query_terms or any(term in word.lower() for term in query_terms):
                start = max(0, i - context_words)
                end = min(len(words), i + context_words + 1)
                highlight = ' '.join(words[start:end])
                if highlight not in highlights:
                    highlights.append(f"...{highlight}...")
                
                if len(highlights) >= 3:
                    break
        
        return highlights
    
    def similar_documents(
        self,
        document_id: str,
        collection: str = "default",
        top_k: int = 5
    ) -> List[SearchResult]:
        """Find documents similar to a given document."""
        # Get the document
        docs = self.db_manager.get([document_id], collection)
        if not docs:
            return []
        
        doc = docs[0]
        
        # Use document content as query
        results = self.semantic_search(doc.content, collection, top_k + 1)
        
        # Remove the query document itself
        results = [r for r in results if r.document_id != document_id]
        
        return results[:top_k]
