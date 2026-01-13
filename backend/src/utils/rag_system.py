"""
rag_system.py

Main RAG (Retrieval-Augmented Generation) system interface.
This module provides a unified interface for the RAG system by combining
all sub-modules: core, document management, and retrieval.

The HealthcareRAGSystem class is the main entry point for using the RAG system.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path

from config.settings import MODELS_DIR
from src.utils.logging_config import get_logger

# Import modular components
from src.utils.rag_core import RAGCore
from src.utils.rag_document_manager import RAGDocumentManager
from src.utils.rag_retriever import RAGRetriever
from src.utils.rag_utils import filter_query_parameters

# Import Document for type hints (needed at runtime for type annotations)
try:
    from langchain_core.documents import Document
except ImportError:
    # Fallback if langchain_core is not available
    Document = Any

logger = get_logger(__name__)


class HealthcareRAGSystem:
    """
    Main RAG system for extracting relevant information from healthcare-related queries.
    
    This class provides a unified interface that combines:
    - Core components (embeddings, vector store)
    - Document management (adding documents)
    - Information retrieval (extracting relevant info)
    
    Usage:
        >>> rag = HealthcareRAGSystem()
        >>> rag.add_texts(["Document 1", "Document 2"])
        >>> result = rag.extract_relevant_info("diabetes in elderly")
        >>> print(result['summary'])
    """
    
    def __init__(
        self,
        vector_store_path: Optional[Path] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        similarity_threshold: float = 0.7,
        top_k: int = 5
    ):
        """
        Initialize the RAG system.
        
        Args:
            vector_store_path: Path to store/load vector database
            embedding_model: Name of the embedding model to use
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks for context preservation
            similarity_threshold: Minimum similarity score for retrieval (currently not used in retriever)
            top_k: Number of top documents to retrieve
            
        Example:
            >>> # Use default settings
            >>> rag = HealthcareRAGSystem()
            
            >>> # Custom configuration
            >>> rag = HealthcareRAGSystem(
            ...     embedding_model="sentence-transformers/all-mpnet-base-v2",
            ...     chunk_size=1500,
            ...     top_k=10
            ... )
        """
        # Initialize core components (embeddings, vectorstore, text splitter)
        self._core = RAGCore(
            vector_store_path=vector_store_path,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k
        )
        
        # Initialize document manager
        self._doc_manager = RAGDocumentManager(
            text_splitter=self._core.text_splitter,
            vectorstore=self._core.vectorstore
        )
        
        # Initialize retriever (pass both retriever and vectorstore for flexibility)
        self._retriever = RAGRetriever(
            retriever=self._core.retriever,
            vectorstore=self._core.vectorstore,
            top_k=top_k
        )
        
        # Store configuration
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        
        logger.info("HealthcareRAGSystem initialized successfully")
    
    # ==================== Document Management Methods ====================
    
    def add_documents(
        self,
        documents: List["Document"],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            metadata: Optional metadata to attach to all documents
            
        Returns:
            Number of document chunks added
        """
        return self._doc_manager.add_documents(documents, metadata)
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Add raw text strings to the vector store.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            Number of document chunks added
        """
        return self._doc_manager.add_texts(texts, metadatas)
    
    def add_pubmed_articles(self, articles: List[Dict[str, Any]]) -> int:
        """
        Add PubMed articles to the vector store.
        
        Args:
            articles: List of article dictionaries with title, abstract, etc.
            
        Returns:
            Number of document chunks added
        """
        return self._doc_manager.add_pubmed_articles(articles)
    
    def add_healthcare_documentation(self, docs_path: Path) -> int:
        """
        Add healthcare documentation files to the vector store.
        
        Args:
            docs_path: Path to documentation directory or file
            
        Returns:
            Number of document chunks added
        """
        return self._doc_manager.add_healthcare_documentation(docs_path)
    
    # ==================== Information Retrieval Methods ====================
    
    def extract_relevant_info(
        self,
        query: str,
        max_docs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract relevant information from the knowledge base based on user query.
        
        Args:
            query: User query/prompt
            max_docs: Maximum number of documents to retrieve (overrides top_k)
            
        Returns:
            Dictionary containing relevant information, sources, and metadata
            
        Example:
            >>> result = rag.extract_relevant_info("diabetes treatment")
            >>> print(result['summary'])
            >>> print(f"Confidence: {result['confidence']}")
        """
        return self._retriever.extract_relevant_info(query, max_docs)
    
    def filter_query_parameters(
        self,
        query: str,
        extracted_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter and extract only relevant parameters from the query.
        
        Args:
            query: Original user query
            extracted_info: Information extracted by RAG system
            
        Returns:
            Dictionary of filtered and relevant parameters
            
        Example:
            >>> info = rag.extract_relevant_info("elderly diabetic patients")
            >>> params = rag.filter_query_parameters("elderly diabetic patients", info)
            >>> print(params['relevant_conditions'])  # ['DIABETES']
        """
        return filter_query_parameters(query, extracted_info)
    
    # ==================== Vector Store Management Methods ====================
    
    def get_vectorstore_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        return self._core.get_vectorstore_stats()
    
    def clear_vectorstore(self):
        """
        Clear all documents from the vector store.
        
        Useful for resetting the knowledge base.
        """
        self._core.clear_vectorstore()
    
    # ==================== Property Accessors ====================
    
    @property
    def embeddings(self):
        """Get the embedding model instance."""
        return self._core.embeddings
    
    @property
    def vectorstore(self):
        """Get the vector store instance."""
        return self._core.vectorstore
    
    @property
    def text_splitter(self):
        """Get the text splitter instance."""
        return self._core.text_splitter
    
    @property
    def retriever(self):
        """Get the retriever instance."""
        return self._core.retriever
