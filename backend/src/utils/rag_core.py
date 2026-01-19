"""
rag_core.py

Core components for the RAG system including embeddings and vector store initialization.
This module handles the foundational setup of the RAG system.
"""

from typing import Optional
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config.settings import MODELS_DIR
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RAGCore:
    """
    Core RAG system components for embeddings and vector store management.
    
    This class handles:
    - Embedding model initialization
    - Vector store creation and loading
    - Text splitting configuration
    - Retriever setup
    """
    
    def __init__(
        self,
        vector_store_path: Optional[Path] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 10  # Increased from 5 for more citations
    ):
        """
        Initialize core RAG components.
        
        Args:
            vector_store_path: Path to store/load vector database
            embedding_model: Name of the embedding model to use
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks for context preservation
            top_k: Number of top documents to retrieve
        """
        self.vector_store_path = vector_store_path or (MODELS_DIR / "rag_vectorstore")
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Core components (initialized by methods below)
        self.embeddings = None
        self.vectorstore = None
        self.text_splitter = None
        self.retriever = None
        
        # Initialize all components
        self._initialize_text_splitter()
        self._initialize_embeddings()
        self._initialize_vectorstore()
        
        logger.info("RAG core components initialized successfully")
    
    def _initialize_text_splitter(self):
        """Initialize the text splitter for document chunking."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Split on paragraphs, lines, sentences, words
        )
        logger.debug(f"Text splitter initialized: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def _initialize_embeddings(self):
        """
        Initialize the embedding model for converting text to vectors.
        
        Uses HuggingFace sentence transformers for efficient semantic embeddings.
        """
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
                encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
            )
            
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise RuntimeError(f"Failed to initialize embedding model: {e}") from e
    
    def _initialize_vectorstore(self):
        """
        Initialize or load the vector store (ChromaDB).
        
        If a vector store exists at the specified path, it will be loaded.
        Otherwise, a new empty vector store will be created.
        """
        try:
            # Check if vector store already exists
            if self.vector_store_path.exists() and any(self.vector_store_path.iterdir()):
                logger.info(f"Loading existing vector store from {self.vector_store_path}")
                self.vectorstore = Chroma(
                    persist_directory=str(self.vector_store_path),
                    embedding_function=self.embeddings
                )
                logger.info("Vector store loaded successfully")
            else:
                logger.info("Creating new vector store")
                # Create empty vector store
                self.vectorstore = Chroma(
                    persist_directory=str(self.vector_store_path),
                    embedding_function=self.embeddings
                )
                logger.info("New vector store created")
            
            # Initialize retriever for document search
            # Use similarity_search_with_score to get relevance scores
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.top_k},
                search_type="similarity"  # Use similarity search for better scoring
            )
            logger.debug(f"Retriever initialized with top_k={self.top_k}")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise RuntimeError(f"Failed to initialize vector store: {e}") from e
    
    def get_vectorstore_stats(self) -> dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary containing:
            - status: Current status of vector store
            - num_documents: Number of documents in the store
            - embedding_model: Name of embedding model
            - chunk_size: Current chunk size setting
            - top_k: Number of documents retrieved per query
            - vector_store_path: Path to the vector store
        """
        try:
            if not self.vectorstore:
                return {
                    'status': 'not_initialized',
                    'num_documents': 0
                }
            
            # Get document count from collection
            try:
                collection = self.vectorstore._collection
                count = collection.count() if collection else 0
            except Exception:
                count = 0
            
            return {
                'status': 'initialized',
                'num_documents': count,
                'embedding_model': self.embedding_model_name,
                'chunk_size': self.chunk_size,
                'top_k': self.top_k,
                'vector_store_path': str(self.vector_store_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def clear_vectorstore(self):
        """
        Clear all documents from the vector store.
        
        This will delete all stored documents but keep the vector store structure.
        Useful for resetting the knowledge base.
        """
        try:
            if not self.vectorstore:
                logger.warning("Vector store not initialized, nothing to clear")
                return
            
            # Try to delete the collection
            try:
                collection = self.vectorstore._collection
                if collection:
                    collection.delete()
                self.vectorstore.persist()
                logger.info("Vector store collection deleted")
            except Exception:
                # If deletion fails, recreate the vectorstore by removing directory
                logger.info("Collection deletion failed, recreating vector store")
                import shutil
                if self.vector_store_path.exists():
                    shutil.rmtree(self.vector_store_path)
                self._initialize_vectorstore()
            
            logger.info("Vector store cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise

