"""
Configuration package.

Provides access to all configuration settings:
- settings: Core application settings
- vector_db_config: Vector database configuration
- embedding_config: Embedding model configuration
"""

from config.settings import *
from config.vector_db_config import (
    VectorDBConfig, VectorDBType, CollectionType,
    vector_db_config, EMBEDDING_DIMENSIONS
)
from config.embedding_config import (
    EmbeddingConfig, EmbeddingProvider, EmbeddingModelType,
    embedding_config, RECOMMENDED_MODELS
)

__all__ = [
    # Vector DB
    'VectorDBConfig', 'VectorDBType', 'CollectionType', 
    'vector_db_config', 'EMBEDDING_DIMENSIONS',
    # Embedding
    'EmbeddingConfig', 'EmbeddingProvider', 'EmbeddingModelType',
    'embedding_config', 'RECOMMENDED_MODELS'
]
