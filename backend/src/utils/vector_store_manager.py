"""
Vector Store Manager.

Unified interface for multiple vector database backends.
Supports ChromaDB, Pinecone, Weaviate, and FAISS.
"""

from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import logging

from config.vector_db_config import (
    VectorDBConfig, VectorDBType, CollectionType, vector_db_config
)
from src.utils.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents with embeddings to the store."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get total document count."""
        pass


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(self, config: VectorDBConfig, collection_name: str = None):
        self.config = config.chroma
        self.collection_name = collection_name or self.config.collection_name
        self._client = None
        self._collection = None
    
    @property
    def client(self):
        if self._client is None:
            import chromadb
            self.config.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.config.persist_directory)
            )
        return self._client
    
    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.config.distance_metric}
            )
        return self._collection
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        import uuid
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas or [{}] * len(texts),
            ids=ids
        )
        
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter
        )
        
        output = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                score = 1.0 - results['distances'][0][i] if results['distances'] else 0.0
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                output.append((doc, score, metadata))
        
        return output
    
    def delete(self, ids: List[str]) -> bool:
        try:
            self.collection.delete(ids=ids)
            return True
        except Exception as e:
            logger.error(f"Error deleting from Chroma: {e}")
            return False
    
    def count(self) -> int:
        return self.collection.count()


class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(self, config: VectorDBConfig, index_name: str = None):
        self.config = config.pinecone
        self.index_name = index_name or self.config.index_name
        self._client = None
        self._index = None
    
    @property
    def client(self):
        if self._client is None:
            from pinecone import Pinecone
            self._client = Pinecone(api_key=self.config.api_key)
        return self._client
    
    @property
    def index(self):
        if self._index is None:
            # Check if index exists, create if not
            existing = [idx.name for idx in self.client.list_indexes()]
            
            if self.index_name not in existing:
                from pinecone import ServerlessSpec
                self.client.create_index(
                    name=self.index_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.config.environment
                    )
                )
            
            self._index = self.client.Index(self.index_name)
        return self._index
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        import uuid
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        vectors = []
        for i, (id_, embedding) in enumerate(zip(ids, embeddings)):
            metadata = metadatas[i] if metadatas else {}
            metadata['text'] = texts[i][:1000]  # Pinecone metadata limit
            vectors.append({
                'id': id_,
                'values': embedding.tolist(),
                'metadata': metadata
            })
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        
        output = []
        for match in results.matches:
            text = match.metadata.pop('text', '')
            output.append((text, match.score, match.metadata))
        
        return output
    
    def delete(self, ids: List[str]) -> bool:
        try:
            self.index.delete(ids=ids)
            return True
        except Exception as e:
            logger.error(f"Error deleting from Pinecone: {e}")
            return False
    
    def count(self) -> int:
        stats = self.index.describe_index_stats()
        return stats.total_vector_count


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation."""
    
    def __init__(self, config: VectorDBConfig, collection_name: str = "default"):
        self.config = config.faiss
        self.collection_name = collection_name
        self._index = None
        self._documents: List[str] = []
        self._metadatas: List[Dict] = []
        self._ids: List[str] = []
        self._index_path = self.config.index_path / f"{collection_name}.faiss"
        self._data_path = self.config.index_path / f"{collection_name}.json"
    
    @property
    def index(self):
        if self._index is None:
            import faiss
            
            self.config.index_path.mkdir(parents=True, exist_ok=True)
            
            if self._index_path.exists():
                self._index = faiss.read_index(str(self._index_path))
                self._load_data()
            else:
                self._index = faiss.IndexFlatIP(self.config.dimension)
        
        return self._index
    
    def _load_data(self):
        import json
        if self._data_path.exists():
            with open(self._data_path, 'r') as f:
                data = json.load(f)
                self._documents = data.get('documents', [])
                self._metadatas = data.get('metadatas', [])
                self._ids = data.get('ids', [])
    
    def _save_data(self):
        import json
        import faiss
        
        faiss.write_index(self.index, str(self._index_path))
        
        with open(self._data_path, 'w') as f:
            json.dump({
                'documents': self._documents,
                'metadatas': self._metadatas,
                'ids': self._ids
            }, f)
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        import uuid
        import faiss
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings.astype('float32'))
        self.index.add(embeddings.astype('float32'))
        
        self._documents.extend(texts)
        self._metadatas.extend(metadatas or [{}] * len(texts))
        self._ids.extend(ids)
        
        self._save_data()
        
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        import faiss
        
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        
        distances, indices = self.index.search(query, top_k)
        
        output = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self._documents):
                output.append((
                    self._documents[idx],
                    float(distances[0][i]),
                    self._metadatas[idx]
                ))
        
        return output
    
    def delete(self, ids: List[str]) -> bool:
        # FAISS doesn't support deletion, would need to rebuild
        logger.warning("FAISS delete requires index rebuild - not implemented")
        return False
    
    def count(self) -> int:
        return self.index.ntotal


class VectorStoreManager:
    """
    Unified vector store manager.
    
    Usage:
        manager = VectorStoreManager()
        manager.add_texts(["doc1", "doc2"], collection=CollectionType.RESEARCH_PAPERS)
        results = manager.search("query", top_k=5)
    """
    
    def __init__(
        self,
        config: Optional[VectorDBConfig] = None,
        embedding_manager: Optional[EmbeddingManager] = None
    ):
        self.config = config or vector_db_config
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self._stores: Dict[str, BaseVectorStore] = {}
    
    def _get_store(self, collection: str = "default") -> BaseVectorStore:
        """Get or create a vector store for the collection."""
        if collection not in self._stores:
            self._stores[collection] = self._create_store(collection)
        return self._stores[collection]
    
    def _create_store(self, collection: str) -> BaseVectorStore:
        """Create a vector store based on active config."""
        stores = {
            VectorDBType.CHROMA: lambda: ChromaVectorStore(self.config, collection),
            VectorDBType.PINECONE: lambda: PineconeVectorStore(self.config, collection),
            VectorDBType.FAISS: lambda: FAISSVectorStore(self.config, collection),
        }
        
        creator = stores.get(self.config.active_db)
        if creator is None:
            raise ValueError(f"Unsupported vector DB: {self.config.active_db}")
        
        logger.info(f"Creating {self.config.active_db.value} store for collection: {collection}")
        return creator()
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        collection: str = "default"
    ) -> List[str]:
        """Add texts to the vector store."""
        if not texts:
            return []
        
        embeddings = self.embedding_manager.embed_texts(texts)
        store = self._get_store(collection)
        
        return store.add_documents(texts, embeddings, metadatas, ids)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        collection: str = "default",
        filter: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        """Search for similar documents."""
        query_embedding = self.embedding_manager.embed_query(query)
        store = self._get_store(collection)
        
        return store.search(query_embedding, top_k, filter)
    
    def search_with_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        collection: str = "default",
        filter: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        """Search using a pre-computed embedding."""
        store = self._get_store(collection)
        return store.search(embedding, top_k, filter)
    
    def delete(self, ids: List[str], collection: str = "default") -> bool:
        """Delete documents by IDs."""
        store = self._get_store(collection)
        return store.delete(ids)
    
    def count(self, collection: str = "default") -> int:
        """Get document count in collection."""
        store = self._get_store(collection)
        return store.count()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats for all collections."""
        stats = {
            "active_db": self.config.active_db.value,
            "embedding_dimension": self.embedding_manager.dimension,
            "collections": {}
        }
        
        for name, store in self._stores.items():
            stats["collections"][name] = {"count": store.count()}
        
        return stats
