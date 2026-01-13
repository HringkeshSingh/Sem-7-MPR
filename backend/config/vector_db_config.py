"""
Vector Database Configuration.

Supports multiple vector databases:
- ChromaDB (default, local)
- Pinecone (cloud, scalable)
- Weaviate (self-hosted or cloud)
- FAISS (local, fast)
"""

import os
from enum import Enum
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


class VectorDBType(Enum):
    """Supported vector database types."""
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    FAISS = "faiss"


class CollectionType(Enum):
    """Collection types for different data categories."""
    RESEARCH_PAPERS = "research_papers"
    CLINICAL_TRIALS = "clinical_trials"
    MEDICAL_NEWS = "medical_news"
    WHO_DATA = "who_data"
    GUIDELINES = "guidelines"
    GENERAL = "general"


@dataclass
class ChromaConfig:
    """ChromaDB configuration."""
    persist_directory: Path = field(default_factory=lambda: Path("models/vectorstore/chroma"))
    collection_name: str = "healthcare_documents"
    distance_metric: str = "cosine"  # cosine, l2, ip


@dataclass
class PineconeConfig:
    """Pinecone configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    environment: str = field(default_factory=lambda: os.getenv("PINECONE_ENVIRONMENT", "us-east-1"))
    index_name: str = "healthcare-rag"
    dimension: int = 384  # Depends on embedding model
    metric: str = "cosine"
    pod_type: str = "p1.x1"
    
    # Collection-specific indexes
    indexes: Dict[str, str] = field(default_factory=lambda: {
        CollectionType.RESEARCH_PAPERS.value: "healthcare-research",
        CollectionType.CLINICAL_TRIALS.value: "healthcare-trials",
        CollectionType.MEDICAL_NEWS.value: "healthcare-news",
        CollectionType.GENERAL.value: "healthcare-general"
    })


@dataclass
class WeaviateConfig:
    """Weaviate configuration."""
    url: str = field(default_factory=lambda: os.getenv("WEAVIATE_URL", "http://localhost:8080"))
    api_key: str = field(default_factory=lambda: os.getenv("WEAVIATE_API_KEY", ""))
    class_name: str = "HealthcareDocument"
    
    # Schema for healthcare documents
    schema: Dict[str, Any] = field(default_factory=lambda: {
        "class": "HealthcareDocument",
        "vectorizer": "none",  # We provide our own vectors
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "title", "dataType": ["string"]},
            {"name": "source", "dataType": ["string"]},
            {"name": "source_id", "dataType": ["string"]},
            {"name": "collection_type", "dataType": ["string"]},
            {"name": "metadata", "dataType": ["object"]}
        ]
    })


@dataclass
class FAISSConfig:
    """FAISS configuration."""
    index_path: Path = field(default_factory=lambda: Path("models/vectorstore/faiss"))
    index_type: str = "IVFFlat"  # Flat, IVFFlat, IVFPQ, HNSW
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 10  # Number of clusters to search
    dimension: int = 384


@dataclass
class VectorDBConfig:
    """Main vector database configuration."""
    # Active database type
    active_db: VectorDBType = field(
        default_factory=lambda: VectorDBType(os.getenv("VECTOR_DB_TYPE", "chroma"))
    )
    
    # Individual DB configs
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    pinecone: PineconeConfig = field(default_factory=PineconeConfig)
    weaviate: WeaviateConfig = field(default_factory=WeaviateConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    
    # Common settings
    batch_size: int = 100  # Batch size for bulk operations
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def get_active_config(self) -> Any:
        """Get configuration for the active database."""
        configs = {
            VectorDBType.CHROMA: self.chroma,
            VectorDBType.PINECONE: self.pinecone,
            VectorDBType.WEAVIATE: self.weaviate,
            VectorDBType.FAISS: self.faiss
        }
        return configs[self.active_db]
    
    @property
    def is_cloud_based(self) -> bool:
        """Check if active DB is cloud-based."""
        return self.active_db in [VectorDBType.PINECONE, VectorDBType.WEAVIATE]


# Dimension mapping for embedding models
EMBEDDING_DIMENSIONS = {
    # Sentence Transformers
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
    
    # Medical domain
    "dmis-lab/biobert-base-cased-v1.2": 768,
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract": 768,
    "emilyalsentzer/Bio_ClinicalBERT": 768,
    
    # OpenAI
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072
}


# Global config instance
vector_db_config = VectorDBConfig()
