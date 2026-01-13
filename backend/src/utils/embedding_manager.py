"""
Embedding Manager.

Unified interface for generating embeddings from multiple providers.
Supports local models (Sentence Transformers, HuggingFace) and APIs (OpenAI).
"""

from typing import List, Optional, Union
import numpy as np
from abc import ABC, abstractmethod
import logging

from config.embedding_config import (
    EmbeddingConfig, EmbeddingProvider, embedding_config
)

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """Sentence Transformers embedding provider."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config.sentence_transformers
        self._model = None
        self._dimension = None
    
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading Sentence Transformer: {self.config.model_name}")
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device
            )
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.show_progress_bar,
            normalize_embeddings=self.config.normalize_embeddings
        )
        return np.array(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
    
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            _ = self.model  # Trigger lazy loading
        return self._dimension


class HuggingFaceProvider(BaseEmbeddingProvider):
    """HuggingFace Transformers provider for medical models."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config.huggingface
        self._model = None
        self._tokenizer = None
        self._dimension = None
    
    def _load_model(self):
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            logger.info(f"Loading HuggingFace model: {self.config.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self._model = AutoModel.from_pretrained(self.config.model_name)
            
            if self.config.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            
            self._dimension = self._model.config.hidden_size
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        import torch
        
        self._load_model()
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            if self.config.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model(**inputs)
            
            # Apply pooling strategy
            if self.config.pooling_strategy == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                embeddings = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            elif self.config.pooling_strategy == "cls":
                embeddings = outputs.last_hidden_state[:, 0]
            else:  # max
                embeddings = torch.max(outputs.last_hidden_state, dim=1)[0]
            
            if self.config.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
    
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension


class OpenAIProvider(BaseEmbeddingProvider):
    """OpenAI API embedding provider."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config.openai
        self._client = None
        self._dimension = None
    
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.config.api_key)
        return self._client
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            response = self.client.embeddings.create(
                model=self.config.model_name,
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        embeddings = np.array(all_embeddings)
        self._dimension = embeddings.shape[1]
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
    
    @property
    def dimension(self) -> int:
        from config.vector_db_config import EMBEDDING_DIMENSIONS
        if self._dimension is None:
            self._dimension = EMBEDDING_DIMENSIONS.get(self.config.model_name, 1536)
        return self._dimension


class EmbeddingManager:
    """
    Unified embedding manager with caching and provider switching.
    
    Usage:
        manager = EmbeddingManager()
        embeddings = manager.embed_texts(["Hello world", "Medical text"])
        query_embedding = manager.embed_query("search query")
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or embedding_config
        self._provider: Optional[BaseEmbeddingProvider] = None
        self._cache: dict = {}
    
    @property
    def provider(self) -> BaseEmbeddingProvider:
        if self._provider is None:
            self._provider = self._create_provider()
        return self._provider
    
    def _create_provider(self) -> BaseEmbeddingProvider:
        """Create the embedding provider based on config."""
        providers = {
            EmbeddingProvider.SENTENCE_TRANSFORMERS: SentenceTransformersProvider,
            EmbeddingProvider.HUGGINGFACE: HuggingFaceProvider,
            EmbeddingProvider.OPENAI: OpenAIProvider,
        }
        
        provider_class = providers.get(self.config.active_provider)
        if provider_class is None:
            raise ValueError(f"Unsupported provider: {self.config.active_provider}")
        
        logger.info(f"Initializing embedding provider: {self.config.active_provider.value}")
        return provider_class(self.config)
    
    def embed_texts(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])
        
        if use_cache and self.config.cache_embeddings:
            return self._embed_with_cache(texts)
        
        return self.provider.embed_texts(texts)
    
    def _embed_with_cache(self, texts: List[str]) -> np.ndarray:
        """Embed texts with caching."""
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = hash(text)
            if cache_key in self._cache:
                results.append((i, self._cache[cache_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if uncached_texts:
            new_embeddings = self.provider.embed_texts(uncached_texts)
            for idx, text, embedding in zip(uncached_indices, uncached_texts, new_embeddings):
                cache_key = hash(text)
                self._cache[cache_key] = embedding
                results.append((idx, embedding))
        
        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results])
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query."""
        return self.provider.embed_query(query)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.provider.dimension
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")
    
    def switch_provider(self, provider: EmbeddingProvider):
        """Switch to a different embedding provider."""
        self.config.active_provider = provider
        self._provider = None
        self.clear_cache()
        logger.info(f"Switched to provider: {provider.value}")
