"""
Vector Database Manager.

Provides unified CRUD operations and batch processing for vector databases.
Supports ChromaDB, Pinecone, FAISS with automatic connection management.
"""

import os
import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
import logging
import json

from config.vector_db_config import (
    VectorDBConfig, VectorDBType, CollectionType, vector_db_config
)

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document to be stored in vector database."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    collection: str = "default"
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "collection": self.collection,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class QueryResult:
    """Result from vector search."""
    document: Document
    score: float
    rank: int


class BaseVectorDBManager(ABC):
    """Abstract base class for vector database managers."""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self._connected = False
        self._collections: Dict[str, Any] = {}
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the database."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection."""
        pass
    
    @abstractmethod
    def create_collection(self, name: str, dimension: int) -> bool:
        """Create a new collection/index."""
        pass
    
    @abstractmethod
    def delete_collection(self, name: str) -> bool:
        """Delete a collection/index."""
        pass
    
    @abstractmethod
    def insert(self, documents: List[Document]) -> List[str]:
        """Insert documents into the database."""
        pass
    
    @abstractmethod
    def update(self, documents: List[Document]) -> int:
        """Update existing documents."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str], collection: str = "default") -> int:
        """Delete documents by IDs."""
        pass
    
    @abstractmethod
    def get(self, ids: List[str], collection: str = "default") -> List[Document]:
        """Get documents by IDs."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        collection: str = "default",
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[QueryResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def count(self, collection: str = "default") -> int:
        """Count documents in collection."""
        pass
    
    @property
    def is_connected(self) -> bool:
        return self._connected


class ChromaDBManager(BaseVectorDBManager):
    """ChromaDB manager implementation."""
    
    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        self._client = None
        self.chroma_config = config.chroma
    
    def connect(self) -> bool:
        """Connect to ChromaDB."""
        try:
            import chromadb
            
            self.chroma_config.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.chroma_config.persist_directory)
            )
            self._connected = True
            logger.info(f"Connected to ChromaDB at {self.chroma_config.persist_directory}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from ChromaDB."""
        self._client = None
        self._collections.clear()
        self._connected = False
        logger.info("Disconnected from ChromaDB")
    
    def _get_collection(self, name: str, dimension: int = None):
        """Get or create a collection."""
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": self.chroma_config.distance_metric}
            )
        return self._collections[name]
    
    def create_collection(self, name: str, dimension: int = None) -> bool:
        """Create a collection."""
        try:
            self._collections[name] = self._client.create_collection(
                name=name,
                metadata={"hnsw:space": self.chroma_config.distance_metric}
            )
            logger.info(f"Created collection: {name}")
            return True
        except Exception as e:
            logger.warning(f"Collection {name} may already exist: {e}")
            self._collections[name] = self._client.get_collection(name)
            return True
    
    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        try:
            self._client.delete_collection(name)
            self._collections.pop(name, None)
            logger.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {name}: {e}")
            return False
    
    def insert(self, documents: List[Document]) -> List[str]:
        """Insert documents with batch processing."""
        if not documents:
            return []
        
        # Group by collection
        by_collection: Dict[str, List[Document]] = {}
        for doc in documents:
            by_collection.setdefault(doc.collection, []).append(doc)
        
        inserted_ids = []
        
        for collection_name, docs in by_collection.items():
            collection = self._get_collection(collection_name)
            
            # Batch insert
            batch_size = self.config.batch_size
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                
                ids = [d.id for d in batch]
                documents_text = [d.content for d in batch]
                embeddings = [d.embedding for d in batch if d.embedding]
                metadatas = [d.metadata for d in batch]
                
                if embeddings and len(embeddings) == len(batch):
                    collection.add(
                        ids=ids,
                        documents=documents_text,
                        embeddings=embeddings,
                        metadatas=metadatas
                    )
                else:
                    collection.add(
                        ids=ids,
                        documents=documents_text,
                        metadatas=metadatas
                    )
                
                inserted_ids.extend(ids)
        
        logger.info(f"Inserted {len(inserted_ids)} documents")
        return inserted_ids
    
    def update(self, documents: List[Document]) -> int:
        """Update existing documents."""
        if not documents:
            return 0
        
        updated = 0
        by_collection: Dict[str, List[Document]] = {}
        for doc in documents:
            by_collection.setdefault(doc.collection, []).append(doc)
        
        for collection_name, docs in by_collection.items():
            collection = self._get_collection(collection_name)
            
            for doc in docs:
                try:
                    update_kwargs = {
                        "ids": [doc.id],
                        "documents": [doc.content],
                        "metadatas": [doc.metadata]
                    }
                    if doc.embedding:
                        update_kwargs["embeddings"] = [doc.embedding]
                    
                    collection.update(**update_kwargs)
                    updated += 1
                except Exception as e:
                    logger.warning(f"Failed to update document {doc.id}: {e}")
        
        return updated
    
    def delete(self, ids: List[str], collection: str = "default") -> int:
        """Delete documents by IDs."""
        try:
            coll = self._get_collection(collection)
            coll.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from {collection}")
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return 0
    
    def get(self, ids: List[str], collection: str = "default") -> List[Document]:
        """Get documents by IDs."""
        try:
            coll = self._get_collection(collection)
            results = coll.get(ids=ids, include=["documents", "metadatas", "embeddings"])
            
            documents = []
            for i, doc_id in enumerate(results["ids"]):
                doc = Document(
                    id=doc_id,
                    content=results["documents"][i] if results["documents"] else "",
                    embedding=results["embeddings"][i] if results.get("embeddings") else None,
                    metadata=results["metadatas"][i] if results["metadatas"] else {},
                    collection=collection
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return []
    
    def search(
        self,
        query_embedding: List[float],
        collection: str = "default",
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[QueryResult]:
        """Search for similar documents."""
        try:
            coll = self._get_collection(collection)
            
            results = coll.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )
            
            query_results = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0
                    score = 1.0 - distance  # Convert distance to similarity
                    
                    doc = Document(
                        id=doc_id,
                        content=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        collection=collection
                    )
                    query_results.append(QueryResult(document=doc, score=score, rank=i + 1))
            
            return query_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def count(self, collection: str = "default") -> int:
        """Count documents in collection."""
        try:
            coll = self._get_collection(collection)
            return coll.count()
        except Exception:
            return 0
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            collections = self._client.list_collections()
            return [c.name for c in collections]
        except Exception:
            return []


class PineconeDBManager(BaseVectorDBManager):
    """Pinecone manager implementation."""
    
    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        self._client = None
        self.pinecone_config = config.pinecone
    
    def connect(self) -> bool:
        """Connect to Pinecone."""
        try:
            from pinecone import Pinecone
            
            if not self.pinecone_config.api_key:
                logger.error("Pinecone API key not configured")
                return False
            
            self._client = Pinecone(api_key=self.pinecone_config.api_key)
            self._connected = True
            logger.info("Connected to Pinecone")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Pinecone."""
        self._client = None
        self._collections.clear()
        self._connected = False
    
    def _get_index(self, name: str):
        """Get or create an index."""
        if name not in self._collections:
            self._collections[name] = self._client.Index(name)
        return self._collections[name]
    
    def create_collection(self, name: str, dimension: int) -> bool:
        """Create a Pinecone index."""
        try:
            from pinecone import ServerlessSpec
            
            existing = [idx.name for idx in self._client.list_indexes()]
            if name not in existing:
                self._client.create_index(
                    name=name,
                    dimension=dimension,
                    metric=self.pinecone_config.metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.pinecone_config.environment
                    )
                )
                logger.info(f"Created Pinecone index: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create Pinecone index: {e}")
            return False
    
    def delete_collection(self, name: str) -> bool:
        """Delete a Pinecone index."""
        try:
            self._client.delete_index(name)
            self._collections.pop(name, None)
            return True
        except Exception as e:
            logger.error(f"Failed to delete Pinecone index: {e}")
            return False
    
    def insert(self, documents: List[Document]) -> List[str]:
        """Insert documents with batch processing."""
        if not documents:
            return []
        
        by_collection: Dict[str, List[Document]] = {}
        for doc in documents:
            by_collection.setdefault(doc.collection, []).append(doc)
        
        inserted_ids = []
        
        for collection_name, docs in by_collection.items():
            index = self._get_index(collection_name)
            
            vectors = []
            for doc in docs:
                if doc.embedding:
                    metadata = doc.metadata.copy()
                    metadata["content"] = doc.content[:1000]  # Pinecone metadata limit
                    vectors.append({
                        "id": doc.id,
                        "values": doc.embedding,
                        "metadata": metadata
                    })
            
            # Batch upsert
            batch_size = self.config.batch_size
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch)
                inserted_ids.extend([v["id"] for v in batch])
        
        return inserted_ids
    
    def update(self, documents: List[Document]) -> int:
        """Update is same as insert for Pinecone (upsert)."""
        ids = self.insert(documents)
        return len(ids)
    
    def delete(self, ids: List[str], collection: str = "default") -> int:
        """Delete documents by IDs."""
        try:
            index = self._get_index(collection)
            index.delete(ids=ids)
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to delete from Pinecone: {e}")
            return 0
    
    def get(self, ids: List[str], collection: str = "default") -> List[Document]:
        """Get documents by IDs."""
        try:
            index = self._get_index(collection)
            results = index.fetch(ids=ids)
            
            documents = []
            for doc_id, data in results.vectors.items():
                doc = Document(
                    id=doc_id,
                    content=data.metadata.get("content", ""),
                    embedding=data.values,
                    metadata=data.metadata,
                    collection=collection
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"Failed to get from Pinecone: {e}")
            return []
    
    def search(
        self,
        query_embedding: List[float],
        collection: str = "default",
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[QueryResult]:
        """Search for similar documents."""
        try:
            index = self._get_index(collection)
            
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filters,
                include_metadata=True
            )
            
            query_results = []
            for i, match in enumerate(results.matches):
                doc = Document(
                    id=match.id,
                    content=match.metadata.get("content", "") if match.metadata else "",
                    metadata=match.metadata or {},
                    collection=collection
                )
                query_results.append(QueryResult(document=doc, score=match.score, rank=i + 1))
            
            return query_results
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []
    
    def count(self, collection: str = "default") -> int:
        """Count documents in index."""
        try:
            index = self._get_index(collection)
            stats = index.describe_index_stats()
            return stats.total_vector_count
        except Exception:
            return 0


class VectorDBManager:
    """
    Unified Vector Database Manager.
    
    Provides a single interface for multiple vector database backends.
    
    Usage:
        manager = VectorDBManager()
        manager.connect()
        
        # Insert documents
        docs = [Document(id="1", content="Medical text...", embedding=[...], collection="research")]
        manager.insert(docs)
        
        # Search
        results = manager.search(query_embedding, collection="research", top_k=10)
        
        # Cleanup
        manager.disconnect()
    """
    
    def __init__(self, config: Optional[VectorDBConfig] = None):
        self.config = config or vector_db_config
        self._manager: Optional[BaseVectorDBManager] = None
    
    def _create_manager(self) -> BaseVectorDBManager:
        """Create appropriate manager based on config."""
        managers = {
            VectorDBType.CHROMA: ChromaDBManager,
            VectorDBType.PINECONE: PineconeDBManager,
        }
        
        manager_class = managers.get(self.config.active_db)
        if manager_class is None:
            logger.warning(f"Unsupported DB type: {self.config.active_db}, falling back to ChromaDB")
            manager_class = ChromaDBManager
        
        return manager_class(self.config)
    
    def connect(self) -> bool:
        """Connect to the configured database."""
        if self._manager is None:
            self._manager = self._create_manager()
        return self._manager.connect()
    
    def disconnect(self) -> None:
        """Disconnect from the database."""
        if self._manager:
            self._manager.disconnect()
    
    @contextmanager
    def session(self):
        """Context manager for database sessions."""
        try:
            self.connect()
            yield self
        finally:
            self.disconnect()
    
    def create_collection(self, name: str, dimension: int = 384) -> bool:
        """Create a new collection."""
        return self._manager.create_collection(name, dimension)
    
    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        return self._manager.delete_collection(name)
    
    def insert(self, documents: List[Document]) -> List[str]:
        """Insert documents."""
        return self._manager.insert(documents)
    
    def insert_batch(
        self,
        contents: List[str],
        embeddings: List[List[float]],
        collection: str = "default",
        metadatas: Optional[List[Dict]] = None
    ) -> List[str]:
        """Convenience method for batch insertion."""
        documents = []
        for i, (content, embedding) in enumerate(zip(contents, embeddings)):
            doc = Document(
                id=str(uuid.uuid4()),
                content=content,
                embedding=embedding,
                metadata=metadatas[i] if metadatas else {},
                collection=collection
            )
            documents.append(doc)
        
        return self.insert(documents)
    
    def update(self, documents: List[Document]) -> int:
        """Update documents."""
        return self._manager.update(documents)
    
    def delete(self, ids: List[str], collection: str = "default") -> int:
        """Delete documents by IDs."""
        return self._manager.delete(ids, collection)
    
    def get(self, ids: List[str], collection: str = "default") -> List[Document]:
        """Get documents by IDs."""
        return self._manager.get(ids, collection)
    
    def search(
        self,
        query_embedding: List[float],
        collection: str = "default",
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[QueryResult]:
        """Search for similar documents."""
        return self._manager.search(query_embedding, collection, top_k, filters)
    
    def count(self, collection: str = "default") -> int:
        """Count documents in collection."""
        return self._manager.count(collection)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            "db_type": self.config.active_db.value,
            "connected": self._manager.is_connected if self._manager else False,
            "collections": {}
        }
        
        if isinstance(self._manager, ChromaDBManager):
            for name in self._manager.list_collections():
                stats["collections"][name] = {"count": self.count(name)}
        
        return stats
    
    @property
    def is_connected(self) -> bool:
        return self._manager.is_connected if self._manager else False
