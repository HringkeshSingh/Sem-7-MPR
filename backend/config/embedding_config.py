"""
Embedding Model Configuration.

Supports multiple embedding providers:
- Local: Sentence Transformers, HuggingFace (BioBERT, PubMedBERT)
- API: OpenAI, Cohere
"""

import os
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    COHERE = "cohere"


class EmbeddingModelType(Enum):
    """Pre-configured embedding model types."""
    # General purpose
    MINILM = "sentence-transformers/all-MiniLM-L6-v2"
    MPNET = "sentence-transformers/all-mpnet-base-v2"
    
    # Medical domain (recommended for healthcare)
    BIOBERT = "dmis-lab/biobert-base-cased-v1.2"
    PUBMEDBERT = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    CLINICALBERT = "emilyalsentzer/Bio_ClinicalBERT"
    
    # OpenAI
    ADA_002 = "text-embedding-ada-002"
    EMBEDDING_3_SMALL = "text-embedding-3-small"
    EMBEDDING_3_LARGE = "text-embedding-3-large"


@dataclass
class SentenceTransformersConfig:
    """Sentence Transformers configuration."""
    model_name: str = EmbeddingModelType.MINILM.value
    device: str = "cpu"  # cpu, cuda, mps
    normalize_embeddings: bool = True
    batch_size: int = 32
    show_progress_bar: bool = False


@dataclass
class HuggingFaceConfig:
    """HuggingFace Transformers configuration for medical models."""
    model_name: str = EmbeddingModelType.PUBMEDBERT.value
    device: str = "cpu"
    max_length: int = 512
    pooling_strategy: str = "mean"  # mean, cls, max
    normalize: bool = True
    batch_size: int = 16


@dataclass
class OpenAIConfig:
    """OpenAI embeddings configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model_name: str = EmbeddingModelType.ADA_002.value
    batch_size: int = 100
    max_retries: int = 3
    timeout: int = 60


@dataclass
class CohereConfig:
    """Cohere embeddings configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))
    model_name: str = "embed-english-v3.0"
    input_type: str = "search_document"  # search_document, search_query


@dataclass
class EmbeddingConfig:
    """Main embedding configuration."""
    # Active provider
    active_provider: EmbeddingProvider = field(
        default_factory=lambda: EmbeddingProvider(
            os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
        )
    )
    
    # Use medical-specific model by default for healthcare
    use_medical_model: bool = True
    
    # Provider configs
    sentence_transformers: SentenceTransformersConfig = field(default_factory=SentenceTransformersConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    cohere: CohereConfig = field(default_factory=CohereConfig)
    
    # Caching
    cache_embeddings: bool = True
    cache_dir: str = "models/embedding_cache"
    
    def get_active_config(self) -> Any:
        """Get configuration for the active provider."""
        configs = {
            EmbeddingProvider.SENTENCE_TRANSFORMERS: self.sentence_transformers,
            EmbeddingProvider.HUGGINGFACE: self.huggingface,
            EmbeddingProvider.OPENAI: self.openai,
            EmbeddingProvider.COHERE: self.cohere
        }
        return configs[self.active_provider]
    
    def get_model_name(self) -> str:
        """Get the active model name."""
        config = self.get_active_config()
        return config.model_name
    
    def get_dimension(self) -> int:
        """Get embedding dimension for active model."""
        from config.vector_db_config import EMBEDDING_DIMENSIONS
        model_name = self.get_model_name()
        return EMBEDDING_DIMENSIONS.get(model_name, 768)
    
    @property
    def is_local(self) -> bool:
        """Check if using local embedding model."""
        return self.active_provider in [
            EmbeddingProvider.SENTENCE_TRANSFORMERS,
            EmbeddingProvider.HUGGINGFACE
        ]
    
    @property
    def is_api_based(self) -> bool:
        """Check if using API-based embedding."""
        return self.active_provider in [
            EmbeddingProvider.OPENAI,
            EmbeddingProvider.COHERE
        ]


# Recommended models for different use cases
RECOMMENDED_MODELS = {
    "general_fast": EmbeddingModelType.MINILM,
    "general_accurate": EmbeddingModelType.MPNET,
    "medical_research": EmbeddingModelType.PUBMEDBERT,
    "medical_clinical": EmbeddingModelType.CLINICALBERT,
    "medical_general": EmbeddingModelType.BIOBERT,
    "api_cost_effective": EmbeddingModelType.ADA_002,
    "api_high_quality": EmbeddingModelType.EMBEDDING_3_LARGE
}


# Global config instance
embedding_config = EmbeddingConfig()
