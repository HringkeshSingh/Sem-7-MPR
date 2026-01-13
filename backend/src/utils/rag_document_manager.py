"""
rag_document_manager.py

Document management for the RAG system.
Handles adding various types of documents to the vector store.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, JSONLoader

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RAGDocumentManager:
    """
    Manages document addition to the RAG vector store.
    
    Supports multiple document types:
    - Raw text strings
    - LangChain Document objects
    - PubMed articles
    - Healthcare documentation files
    """
    
    def __init__(self, text_splitter, vectorstore):
        """
        Initialize document manager.
        
        Args:
            text_splitter: Text splitter for chunking documents
            vectorstore: Vector store instance for storing documents
        """
        self.text_splitter = text_splitter
        self.vectorstore = vectorstore
    
    def add_documents(
        self,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add LangChain Document objects to the vector store.
        
        Documents are automatically split into chunks before being added.
        
        Args:
            documents: List of Document objects to add
            metadata: Optional metadata to attach to all documents
            
        Returns:
            Number of document chunks added
            
        Raises:
            ValueError: If no documents provided
            RuntimeError: If addition fails
        """
        if not documents:
            logger.warning("No documents provided to add")
            return 0
        
        try:
            # Split documents into chunks for better retrieval
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            
            # Add metadata to all chunks if provided
            if metadata:
                for chunk in chunks:
                    chunk.metadata.update(metadata)
            
            # Add chunks to vector store
            self.vectorstore.add_documents(chunks)
            self.vectorstore.persist()  # Save to disk
            
            logger.info(f"Successfully added {len(chunks)} document chunks to vector store")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise RuntimeError(f"Failed to add documents: {e}") from e
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Add raw text strings to the vector store.
        
        Converts text strings to Document objects before processing.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries (one per text)
            
        Returns:
            Number of document chunks added
        """
        try:
            # Ensure metadatas list matches texts list
            if metadatas is None:
                metadatas = [{}] * len(texts)
            elif len(metadatas) != len(texts):
                logger.warning(f"Metadata count ({len(metadatas)}) doesn't match text count ({len(texts)}). Using empty metadata for missing items.")
                metadatas.extend([{}] * (len(texts) - len(metadatas)))
            
            # Convert texts to Document objects
            documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(texts, metadatas)
            ]
            
            # Add using the document method
            return self.add_documents(documents)
            
        except Exception as e:
            logger.error(f"Error adding texts: {e}")
            raise
    
    def add_pubmed_articles(self, articles: List[Dict[str, Any]]) -> int:
        """
        Add PubMed articles to the vector store.
        
        Combines title and abstract for better retrieval. Preserves article metadata
        including PMID, journal, authors, etc.
        
        Args:
            articles: List of article dictionaries with keys:
                - title: Article title
                - abstract: Article abstract
                - pmid: PubMed ID
                - journal: Journal name
                - year: Publication year
                - authors: List of author names
                - doi: DOI (optional)
                - relevance_score: Relevance score (optional)
                
        Returns:
            Number of document chunks added
        """
        if not articles:
            logger.warning("No PubMed articles provided")
            return 0
        
        try:
            documents = []
            
            for article in articles:
                # Combine title and abstract for comprehensive retrieval
                title = article.get('title', 'No title')
                abstract = article.get('abstract', 'No abstract available')
                content = f"Title: {title}\n\nAbstract: {abstract}"
                
                # Preserve article metadata
                metadata = {
                    'source': 'pubmed',
                    'pmid': article.get('pmid', ''),
                    'journal': article.get('journal', ''),
                    'year': article.get('year', ''),
                    'authors': ', '.join(article.get('authors', [])),
                    'doi': article.get('doi', ''),
                    'relevance_score': article.get('relevance_score', 0.0)
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
            
            chunks_added = self.add_documents(documents)
            logger.info(f"Added {len(articles)} PubMed articles ({chunks_added} chunks) to vector store")
            return chunks_added
            
        except Exception as e:
            logger.error(f"Error adding PubMed articles: {e}")
            raise
    
    def add_healthcare_documentation(self, docs_path: Path) -> int:
        """
        Add healthcare documentation files to the vector store.
        
        Supports markdown (.md), text (.txt), and JSON (.json) files.
        Recursively searches directories for documentation files.
        
        Args:
            docs_path: Path to documentation directory or single file
            
        Returns:
            Number of document chunks added
        """
        if not docs_path.exists():
            logger.warning(f"Documentation path does not exist: {docs_path}")
            return 0
        
        try:
            documents = []
            
            # Determine files to process
            if docs_path.is_file():
                files = [docs_path]
            else:
                # Find all markdown, text, and JSON files recursively
                files = (
                    list(docs_path.glob("**/*.md")) +
                    list(docs_path.glob("**/*.txt")) +
                    list(docs_path.glob("**/*.json"))
                )
            
            logger.info(f"Found {len(files)} documentation files to process")
            
            # Load each file
            for file_path in files:
                try:
                    # Choose appropriate loader based on file type
                    if file_path.suffix == '.json':
                        loader = JSONLoader(file_path=str(file_path), jq_schema=".")
                    else:
                        loader = TextLoader(file_path=str(file_path))
                    
                    # Load documents from file
                    loaded_docs = loader.load()
                    
                    # Add file metadata to each document
                    for doc in loaded_docs:
                        doc.metadata['source'] = str(file_path)
                        doc.metadata['file_type'] = file_path.suffix
                        doc.metadata['file_name'] = file_path.name
                    
                    documents.extend(loaded_docs)
                    logger.debug(f"Loaded {len(loaded_docs)} documents from {file_path.name}")
                    
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
                    continue  # Skip problematic files but continue processing
            
            # Add all loaded documents
            if documents:
                chunks_added = self.add_documents(documents)
                logger.info(f"Added {len(documents)} documentation files ({chunks_added} chunks) to vector store")
                return chunks_added
            else:
                logger.warning("No documents found to add")
                return 0
                
        except Exception as e:
            logger.error(f"Error adding healthcare documentation: {e}")
            raise

