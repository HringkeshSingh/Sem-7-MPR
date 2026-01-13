"""
rag_retriever.py

Information retrieval and extraction for the RAG system.
Handles query processing and relevant document retrieval.
"""

from typing import List, Dict, Any, Optional

from src.utils.logging_config import get_logger
from src.utils.rag_utils import generate_summary, calculate_confidence

logger = get_logger(__name__)


class RAGRetriever:
    """
    Handles information retrieval from the RAG vector store.
    
    Processes user queries and extracts relevant information from
    the knowledge base using semantic search.
    """
    
    def __init__(self, retriever, vectorstore=None, top_k: int = 5):
        """
        Initialize retriever.
        
        Args:
            retriever: LangChain retriever instance (may be None if using vectorstore directly)
            vectorstore: Vector store instance (for direct similarity search)
            top_k: Default number of documents to retrieve
        """
        self.retriever = retriever
        self.vectorstore = vectorstore
        self.top_k = top_k
        
        # Log initialization for debugging
        logger.debug(f"RAGRetriever initialized: vectorstore={vectorstore is not None}, retriever={retriever is not None}, top_k={top_k}")
    
    def extract_relevant_info(
        self,
        query: str,
        max_docs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract relevant information from the knowledge base based on user query.
        
        Performs semantic search to find the most relevant documents, then
        processes them into a structured response with summary and metadata.
        
        Args:
            query: User query/prompt
            max_docs: Maximum number of documents to retrieve (overrides top_k)
            
        Returns:
            Dictionary containing:
            - relevant_info: List of relevant document information
            - summary: Human-readable summary of findings
            - sources: List of source metadata
            - confidence: Confidence score (0-1)
            - query: Original query
            - num_documents: Number of documents retrieved
        """
        try:
            # Determine number of documents to retrieve
            k = max_docs or self.top_k
            
            # ALWAYS use vectorstore directly first - this bypasses retriever API issues
            docs = None
            retrieval_error = None
            
            # Method 1: Use vectorstore similarity_search_with_score directly (MOST RELIABLE)
            # This completely bypasses the retriever wrapper to avoid LangChain API compatibility issues
            if self.vectorstore:
                try:
                    logger.debug(f"Attempting vectorstore similarity_search_with_score for query: {query[:50]}...")
                    
                    # Check if vectorstore has any documents first
                    try:
                        # Try to get collection count to check if vectorstore has documents
                        if hasattr(self.vectorstore, '_collection'):
                            collection = self.vectorstore._collection
                            if collection:
                                count = collection.count()
                                logger.debug(f"Vectorstore has {count} documents")
                                if count == 0:
                                    logger.warning("Vectorstore is empty - no documents to search")
                                    docs = []
                                    # Don't raise error, just return empty results
                    except Exception as count_error:
                        logger.debug(f"Could not check vectorstore count: {count_error}")
                        # Continue anyway - might still have documents
                    
                    # Try similarity_search_with_score first (returns tuples with scores)
                    if hasattr(self.vectorstore, 'similarity_search_with_score'):
                        # IMPORTANT: Call the method directly, don't go through any wrapper
                        # This ensures we bypass any retriever that might be created internally
                        method = getattr(self.vectorstore, 'similarity_search_with_score')
                        results = method(query, k=k)
                        # Results are (Document, score) tuples
                        docs = []
                        for doc, score in results:
                            # Store score in document metadata
                            if hasattr(doc, 'metadata'):
                                if not isinstance(doc.metadata, dict):
                                    doc.metadata = {}
                                doc.metadata['_retrieval_score'] = float(score)
                            docs.append(doc)
                        logger.info(f"✅ Successfully retrieved {len(docs)} documents using similarity_search_with_score")
                    # Fallback to regular similarity_search if with_score not available
                    elif hasattr(self.vectorstore, 'similarity_search'):
                        docs = self.vectorstore.similarity_search(query, k=k)
                        logger.info(f"✅ Successfully retrieved {len(docs)} documents using similarity_search")
                    else:
                        raise AttributeError("Vectorstore does not have similarity_search methods")
                except Exception as e:
                    retrieval_error = e
                    error_str = str(e)
                    logger.error(f"❌ Vectorstore search failed: {error_str}")
                    
                    # If error mentions get_relevant_documents, it means something is calling the retriever
                    if 'get_relevant_documents' in error_str:
                        logger.error("⚠️ ERROR: Something is still trying to use get_relevant_documents!")
                        logger.error("This should not happen - we're using vectorstore directly")
                        # Return empty response instead of raising
                        return self._empty_response(query, "Vectorstore search error - please check backend logs")
                    
                    logger.exception("Full traceback:")
                    docs = None
            else:
                logger.warning("⚠️ Vectorstore not available, will try retriever methods")
            
            # Method 2: Try retriever invoke() method (LangChain 0.1.0+) - ONLY if vectorstore is completely unavailable
            # NEVER use retriever if we have a vectorstore - it may internally call get_relevant_documents
            if docs is None and self.retriever and not self.vectorstore:
                # Only try retriever if we absolutely don't have a vectorstore
                logger.warning("⚠️ No vectorstore available, attempting retriever (may fail with get_relevant_documents error)")
                try:
                    if hasattr(self.retriever, 'invoke'):
                        result = self.retriever.invoke(query)
                        if result and len(result) > 0:
                            if isinstance(result[0], tuple):
                                docs = []
                                for doc, score in result:
                                    if hasattr(doc, 'metadata'):
                                        if not isinstance(doc.metadata, dict):
                                            doc.metadata = {}
                                        doc.metadata['_retrieval_score'] = float(score)
                                    docs.append(doc)
                            else:
                                docs = result if result else []
                        logger.info(f"Successfully retrieved {len(docs) if docs else 0} documents using invoke()")
                except AttributeError as e:
                    # Specifically catch AttributeError for get_relevant_documents
                    if 'get_relevant_documents' in str(e):
                        logger.error("❌ Retriever tried to use get_relevant_documents - this is a LangChain version incompatibility")
                        logger.error("Solution: Ensure vectorstore is properly initialized and has documents")
                        raise RuntimeError("Retriever API incompatibility. Please ensure vectorstore is available and has documents loaded.")
                    raise
                except Exception as e:
                    if retrieval_error is None:
                        retrieval_error = e
                    logger.warning(f"invoke() failed: {e}")
            
            # If all methods failed, raise an error
            if docs is None:
                if not self.retriever and not self.vectorstore:
                    error_msg = "Neither retriever nor vectorstore is available"
                else:
                    error_msg = f"All retrieval methods failed. Last error: {retrieval_error or 'Unknown error'}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Handle case where docs is empty list (no documents found)
            if docs is not None and len(docs) == 0:
                logger.info(f"No relevant documents found for query: {query}")
                return self._empty_response(query, "No relevant information found in the knowledge base. The vectorstore may be empty or the query doesn't match any documents.")
            
            # If docs is still None at this point, we had an error
            if docs is None:
                # This should have been handled above, but just in case
                error_detail = retrieval_error or "Unknown error"
                if 'get_relevant_documents' in str(error_detail):
                    return self._empty_response(query, "RAG system error: LangChain API incompatibility. Please check backend logs and ensure vectorstore has documents loaded.")
                return self._empty_response(query, f"Error retrieving documents: {error_detail}")
            
            # Process retrieved documents
            relevant_info, sources = self._process_documents(docs)
            
            # Generate summary and calculate confidence
            summary = generate_summary(query, relevant_info)
            confidence = calculate_confidence(relevant_info)
            
            logger.info(f"Extracted {len(relevant_info)} relevant documents for query: {query}")
            
            return {
                'relevant_info': relevant_info,
                'summary': summary,
                'sources': sources,
                'confidence': confidence,
                'query': query,
                'num_documents': len(relevant_info)
            }
            
        except Exception as e:
            logger.error(f"Error extracting relevant information: {e}")
            return self._empty_response(query, f"Error extracting information: {str(e)}")
    
    def _process_documents(self, docs: List) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process retrieved documents into structured format.
        
        Args:
            docs: List of retrieved Document objects
            
        Returns:
            Tuple of (relevant_info, sources) lists
        """
        relevant_info = []
        sources = []
        
        for i, doc in enumerate(docs):
            try:
                # Handle different document formats
                # Standard LangChain Document has page_content and metadata
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    metadata = getattr(doc, 'metadata', {})
                    # Try to get score from document or metadata
                    score = getattr(doc, 'score', None)
                    if score is None and isinstance(metadata, dict):
                        score = metadata.get('score', None)
                    # If still no score, calculate a default based on position (higher = more relevant)
                    if score is None:
                        # Use inverse position as score (first doc = highest score)
                        score = max(0.0, 1.0 - (i * 0.1))
                elif isinstance(doc, dict):
                    # Document might be a dictionary
                    content = doc.get('page_content', doc.get('content', str(doc)))
                    metadata = doc.get('metadata', {})
                    score = doc.get('score', metadata.get('score', max(0.0, 1.0 - (i * 0.1))))
                else:
                    # Fallback: try to convert to string
                    content = str(doc)
                    metadata = {}
                    score = max(0.0, 1.0 - (i * 0.1))
                
                # Extract relevance score - prioritize retrieval score, then metadata score, then position-based
                relevance_score = 0.0
                if isinstance(metadata, dict):
                    # Check for retrieval score from similarity search
                    if '_retrieval_score' in metadata:
                        # Normalize similarity score (usually 0-1 or distance-based)
                        raw_score = metadata['_retrieval_score']
                        # If it's a distance (lower is better), convert to similarity (higher is better)
                        if raw_score > 1.0:
                            # Likely a distance metric, invert it
                            relevance_score = 1.0 / (1.0 + raw_score)
                        else:
                            relevance_score = min(max(float(raw_score), 0.0), 1.0)
                    elif 'score' in metadata:
                        relevance_score = float(metadata['score'])
                
                # If still no score, use position-based (first = highest)
                if relevance_score == 0.0:
                    # First document gets 0.9, decreasing by 0.1 for each subsequent
                    relevance_score = max(0.3, 1.0 - (i * 0.15))
                
                # Extract document information
                info = {
                    'content': content,
                    'metadata': metadata if isinstance(metadata, dict) else {},
                    'relevance_score': relevance_score
                }
                relevant_info.append(info)
                
                # Extract source information for citation
                if isinstance(metadata, dict):
                    source_info = {
                        'source': metadata.get('source', 'unknown'),
                        'pmid': metadata.get('pmid', ''),
                        'journal': metadata.get('journal', ''),
                        'year': metadata.get('year', '')
                    }
                else:
                    source_info = {
                        'source': 'unknown',
                        'pmid': '',
                        'journal': '',
                        'year': ''
                    }
                sources.append(source_info)
                
            except Exception as e:
                logger.warning(f"Error processing document: {e}, skipping")
                continue
        
        return relevant_info, sources
    
    def _empty_response(self, query: str, message: str) -> Dict[str, Any]:
        """
        Create an empty response structure.
        
        Args:
            query: Original query
            message: Message explaining why response is empty
            
        Returns:
            Empty response dictionary
        """
        return {
            'relevant_info': [],
            'summary': message,
            'sources': [],
            'confidence': 0.0,
            'query': query,
            'num_documents': 0
        }

