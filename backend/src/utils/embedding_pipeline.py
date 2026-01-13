"""
Embedding Pipeline.

Handles document chunking, embedding generation, and rate limiting.
Supports multiple embedding providers with automatic fallback.
"""

import os
import time
import asyncio
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from config.embedding_config import (
    EmbeddingConfig, EmbeddingProvider, embedding_config
)

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Document chunk with metadata."""
    id: str
    content: str
    index: int
    total_chunks: int
    parent_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "index": self.index,
            "total_chunks": self.total_chunks,
            "parent_id": self.parent_id,
            "metadata": self.metadata
        }


class TextChunker:
    """
    Intelligent text chunker for medical documents.
    
    Supports multiple chunking strategies:
    - Fixed size with overlap
    - Sentence-based
    - Paragraph-based
    - Semantic (section-based)
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "fixed"  # fixed, sentence, paragraph, semantic
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
    
    def chunk(
        self,
        text: str,
        parent_id: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """Chunk text using configured strategy."""
        if self.strategy == "sentence":
            return self._chunk_by_sentence(text, parent_id, metadata)
        elif self.strategy == "paragraph":
            return self._chunk_by_paragraph(text, parent_id, metadata)
        elif self.strategy == "semantic":
            return self._chunk_semantic(text, parent_id, metadata)
        else:
            return self._chunk_fixed(text, parent_id, metadata)
    
    def _chunk_fixed(
        self,
        text: str,
        parent_id: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """Fixed-size chunking with overlap."""
        chunks = []
        text = text.strip()
        
        if len(text) <= self.chunk_size:
            chunks.append(Chunk(
                id=self._generate_chunk_id(parent_id, 0),
                content=text,
                index=0,
                total_chunks=1,
                parent_id=parent_id,
                metadata=metadata or {}
            ))
            return chunks
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Find a good breaking point
            if end < len(text):
                # Try to break at sentence end
                for sep in ['. ', '.\n', '! ', '? ', '\n\n']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    id=self._generate_chunk_id(parent_id, chunk_index),
                    content=chunk_text,
                    index=chunk_index,
                    total_chunks=0,  # Will be updated
                    parent_id=parent_id,
                    metadata=metadata or {}
                ))
                chunk_index += 1
            
            start = end - self.chunk_overlap
        
        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_by_sentence(
        self,
        text: str,
        parent_id: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """Chunk by sentences, grouping to target size."""
        import re
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_length + len(sentence) > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    id=self._generate_chunk_id(parent_id, chunk_index),
                    content=chunk_text,
                    index=chunk_index,
                    total_chunks=0,
                    parent_id=parent_id,
                    metadata=metadata or {}
                ))
                chunk_index += 1
                
                # Keep overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s)
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += len(sentence)
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                id=self._generate_chunk_id(parent_id, chunk_index),
                content=chunk_text,
                index=chunk_index,
                total_chunks=0,
                parent_id=parent_id,
                metadata=metadata or {}
            ))
        
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_by_paragraph(
        self,
        text: str,
        parent_id: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """Chunk by paragraphs."""
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if current_length + len(para) > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(Chunk(
                    id=self._generate_chunk_id(parent_id, chunk_index),
                    content=chunk_text,
                    index=chunk_index,
                    total_chunks=0,
                    parent_id=parent_id,
                    metadata=metadata or {}
                ))
                chunk_index += 1
                current_chunk = []
                current_length = 0
            
            current_chunk.append(para)
            current_length += len(para)
        
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(Chunk(
                id=self._generate_chunk_id(parent_id, chunk_index),
                content=chunk_text,
                index=chunk_index,
                total_chunks=0,
                parent_id=parent_id,
                metadata=metadata or {}
            ))
        
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_semantic(
        self,
        text: str,
        parent_id: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """Chunk by semantic sections (headers, etc.)."""
        import re
        
        # Look for section headers
        section_pattern = r'\n(?=[A-Z][^a-z]*:|\d+\.\s|#{1,6}\s|[A-Z]{2,})'
        sections = re.split(section_pattern, text)
        
        if len(sections) <= 1:
            return self._chunk_fixed(text, parent_id, metadata)
        
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If section is too long, use fixed chunking
            if len(section) > self.chunk_size * 2:
                sub_chunks = self._chunk_fixed(section, f"{parent_id}_s{chunk_index}", metadata)
                for sub in sub_chunks:
                    sub.index = chunk_index
                    sub.id = self._generate_chunk_id(parent_id, chunk_index)
                    chunks.append(sub)
                    chunk_index += 1
            else:
                chunks.append(Chunk(
                    id=self._generate_chunk_id(parent_id, chunk_index),
                    content=section,
                    index=chunk_index,
                    total_chunks=0,
                    parent_id=parent_id,
                    metadata=metadata or {}
                ))
                chunk_index += 1
        
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _generate_chunk_id(self, parent_id: str, index: int) -> str:
        """Generate unique chunk ID."""
        return f"{parent_id}_chunk_{index}"


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 150000
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._request_times: List[float] = []
        self._token_counts: List[Tuple[float, int]] = []
    
    def wait_if_needed(self, estimated_tokens: int = 0):
        """Block if rate limit would be exceeded."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old entries
        self._request_times = [t for t in self._request_times if t > minute_ago]
        self._token_counts = [(t, c) for t, c in self._token_counts if t > minute_ago]
        
        # Check request limit
        if len(self._request_times) >= self.requests_per_minute:
            sleep_time = self._request_times[0] - minute_ago
            if sleep_time > 0:
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        # Check token limit
        total_tokens = sum(c for _, c in self._token_counts) + estimated_tokens
        if total_tokens >= self.tokens_per_minute:
            sleep_time = self._token_counts[0][0] - minute_ago
            if sleep_time > 0:
                logger.debug(f"Token rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        # Record this request
        self._request_times.append(current_time)
        if estimated_tokens > 0:
            self._token_counts.append((current_time, estimated_tokens))


class EmbeddingPipeline:
    """
    Complete embedding pipeline with chunking, rate limiting, and caching.
    
    Usage:
        pipeline = EmbeddingPipeline()
        
        # Embed single text
        embedding = pipeline.embed_text("Medical document content...")
        
        # Process documents with chunking
        chunks = pipeline.process_document(
            "Long document content...",
            doc_id="doc_123",
            chunk_strategy="sentence"
        )
        
        # Batch embed
        embeddings = pipeline.embed_batch(["text1", "text2", "text3"])
    """
    
    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunk_strategy: str = "fixed",
        enable_cache: bool = True
    ):
        self.config = config or embedding_config
        self.chunker = TextChunker(chunk_size, chunk_overlap, chunk_strategy)
        self.rate_limiter = RateLimiter()
        self.enable_cache = enable_cache
        self._cache: Dict[str, List[float]] = {}
        self._model = None
    
    def _get_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            if self.config.active_provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.sentence_transformers.model_name
                device = self.config.sentence_transformers.device
                logger.info(f"Loading embedding model: {model_name}")
                self._model = SentenceTransformer(model_name, device=device)
            elif self.config.active_provider == EmbeddingProvider.OPENAI:
                from openai import OpenAI
                self._model = OpenAI(api_key=self.config.openai.api_key)
            else:
                raise ValueError(f"Unsupported provider: {self.config.active_provider}")
        return self._model
    
    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """Embed a single text."""
        if use_cache and self.enable_cache:
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        embedding = self._generate_embedding([text])[0]
        
        if use_cache and self.enable_cache:
            self._cache[cache_key] = embedding
        
        return embedding
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """Embed multiple texts with batching."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Check cache
            cached = []
            to_embed = []
            cache_indices = []
            
            for j, text in enumerate(batch):
                cache_key = self._cache_key(text)
                if self.enable_cache and cache_key in self._cache:
                    cached.append((j, self._cache[cache_key]))
                else:
                    to_embed.append(text)
                    cache_indices.append(j)
            
            # Embed uncached texts
            new_embeddings = []
            if to_embed:
                self.rate_limiter.wait_if_needed(len(to_embed) * 100)
                new_embeddings = self._generate_embedding(to_embed)
                
                # Cache new embeddings
                for text, emb in zip(to_embed, new_embeddings):
                    if self.enable_cache:
                        self._cache[self._cache_key(text)] = emb
            
            # Merge cached and new embeddings
            batch_embeddings = [None] * len(batch)
            for j, emb in cached:
                batch_embeddings[j] = emb
            
            emb_idx = 0
            for j in cache_indices:
                if emb_idx < len(new_embeddings):
                    batch_embeddings[j] = new_embeddings[emb_idx]
                    emb_idx += 1
            
            all_embeddings.extend(batch_embeddings)
            
            if show_progress:
                logger.info(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} texts")
        
        return all_embeddings
    
    def _generate_embedding(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using configured provider."""
        model = self._get_model()
        
        if self.config.active_provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            embeddings = model.encode(
                texts,
                normalize_embeddings=self.config.sentence_transformers.normalize_embeddings,
                show_progress_bar=False
            )
            return embeddings.tolist()
        
        elif self.config.active_provider == EmbeddingProvider.OPENAI:
            response = model.embeddings.create(
                model=self.config.openai.model_name,
                input=texts
            )
            return [item.embedding for item in response.data]
        
        else:
            raise ValueError(f"Unsupported provider: {self.config.active_provider}")
    
    def process_document(
        self,
        content: str,
        doc_id: str,
        metadata: Optional[Dict] = None,
        chunk_strategy: Optional[str] = None
    ) -> List[Chunk]:
        """
        Process a document: chunk and embed.
        
        Args:
            content: Document content
            doc_id: Unique document ID
            metadata: Optional metadata
            chunk_strategy: Override default chunking strategy
        """
        # Override strategy if provided
        if chunk_strategy:
            original_strategy = self.chunker.strategy
            self.chunker.strategy = chunk_strategy
        
        # Chunk document
        chunks = self.chunker.chunk(content, doc_id, metadata)
        
        # Restore strategy
        if chunk_strategy:
            self.chunker.strategy = original_strategy
        
        # Embed chunks
        if chunks:
            texts = [c.content for c in chunks]
            embeddings = self.embed_batch(texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
        
        logger.info(f"Processed document {doc_id}: {len(chunks)} chunks")
        return chunks
    
    def process_documents(
        self,
        documents: List[Tuple[str, str, Optional[Dict]]],
        chunk_strategy: Optional[str] = None
    ) -> List[Chunk]:
        """Process multiple documents."""
        all_chunks = []
        
        for content, doc_id, metadata in documents:
            chunks = self.process_document(content, doc_id, metadata, chunk_strategy)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.config.get_dimension()
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_embeddings": len(self._cache),
            "cache_enabled": self.enable_cache
        }
