"""
Knowledge Base Updater.

Handles periodic retrieval, indexing, and versioning of the knowledge base.
Supports incremental updates and scheduled refreshes.
"""

import os
import json
import time
import asyncio
import threading
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
import schedule

from src.utils.vector_db_manager import VectorDBManager, Document
from src.utils.embedding_pipeline import EmbeddingPipeline
from src.utils.content_processor import ContentProcessor

logger = logging.getLogger(__name__)


class UpdateStatus(Enum):
    """Update job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class UpdateJob:
    """Represents an update job."""
    id: str
    source: str
    status: UpdateStatus = UpdateStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    documents_processed: int = 0
    documents_added: int = 0
    documents_updated: int = 0
    documents_skipped: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "documents_processed": self.documents_processed,
            "documents_added": self.documents_added,
            "documents_updated": self.documents_updated,
            "documents_skipped": self.documents_skipped,
            "error": self.error
        }


@dataclass
class KnowledgeVersion:
    """Knowledge base version info."""
    version: int
    created_at: datetime
    documents_count: int
    sources: List[str]
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "documents_count": self.documents_count,
            "sources": self.sources,
            "checksum": self.checksum,
            "metadata": self.metadata
        }


class DocumentTracker:
    """Tracks document state for incremental updates."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._state_file = storage_path / "document_state.json"
        self._state: Dict[str, Dict[str, Any]] = self._load_state()
    
    def _load_state(self) -> Dict[str, Dict[str, Any]]:
        """Load document state from disk."""
        if self._state_file.exists():
            try:
                with open(self._state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load document state: {e}")
        return {}
    
    def _save_state(self):
        """Save document state to disk."""
        try:
            with open(self._state_file, 'w') as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save document state: {e}")
    
    def compute_hash(self, content: str) -> str:
        """Compute content hash."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_new_or_updated(self, doc_id: str, content: str) -> bool:
        """Check if document is new or has been updated."""
        content_hash = self.compute_hash(content)
        
        if doc_id not in self._state:
            return True
        
        return self._state[doc_id].get("hash") != content_hash
    
    def mark_processed(self, doc_id: str, content: str, source: str):
        """Mark document as processed."""
        self._state[doc_id] = {
            "hash": self.compute_hash(content),
            "source": source,
            "processed_at": datetime.now().isoformat()
        }
        self._save_state()
    
    def get_processed_ids(self, source: Optional[str] = None) -> Set[str]:
        """Get IDs of processed documents."""
        if source:
            return {doc_id for doc_id, state in self._state.items() 
                   if state.get("source") == source}
        return set(self._state.keys())
    
    def clear(self, source: Optional[str] = None):
        """Clear tracking state."""
        if source:
            self._state = {k: v for k, v in self._state.items() 
                         if v.get("source") != source}
        else:
            self._state = {}
        self._save_state()


class VersionManager:
    """Manages knowledge base versions."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._versions_file = storage_path / "versions.json"
        self._versions: List[KnowledgeVersion] = self._load_versions()
    
    def _load_versions(self) -> List[KnowledgeVersion]:
        """Load version history."""
        if self._versions_file.exists():
            try:
                with open(self._versions_file, 'r') as f:
                    data = json.load(f)
                    return [KnowledgeVersion(
                        version=v["version"],
                        created_at=datetime.fromisoformat(v["created_at"]),
                        documents_count=v["documents_count"],
                        sources=v["sources"],
                        checksum=v["checksum"],
                        metadata=v.get("metadata", {})
                    ) for v in data]
            except Exception as e:
                logger.error(f"Failed to load versions: {e}")
        return []
    
    def _save_versions(self):
        """Save version history."""
        try:
            with open(self._versions_file, 'w') as f:
                json.dump([v.to_dict() for v in self._versions], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save versions: {e}")
    
    def create_version(
        self,
        documents_count: int,
        sources: List[str],
        metadata: Optional[Dict] = None
    ) -> KnowledgeVersion:
        """Create a new version."""
        next_version = len(self._versions) + 1
        
        # Compute checksum from sources and count
        checksum_data = f"{next_version}:{documents_count}:{','.join(sorted(sources))}"
        checksum = hashlib.md5(checksum_data.encode()).hexdigest()[:8]
        
        version = KnowledgeVersion(
            version=next_version,
            created_at=datetime.now(),
            documents_count=documents_count,
            sources=sources,
            checksum=checksum,
            metadata=metadata or {}
        )
        
        self._versions.append(version)
        self._save_versions()
        
        logger.info(f"Created knowledge base version {next_version}")
        return version
    
    def get_current_version(self) -> Optional[KnowledgeVersion]:
        """Get the current (latest) version."""
        return self._versions[-1] if self._versions else None
    
    def get_version_history(self) -> List[KnowledgeVersion]:
        """Get full version history."""
        return self._versions
    
    def rollback_to(self, version: int) -> bool:
        """Rollback to a specific version (marks later versions as invalid)."""
        if version > len(self._versions):
            return False
        
        self._versions = self._versions[:version]
        self._save_versions()
        logger.info(f"Rolled back to version {version}")
        return True


class KnowledgeUpdater:
    """
    Knowledge Base Updater with scheduling and incremental updates.
    
    Usage:
        updater = KnowledgeUpdater()
        
        # Register data sources
        updater.register_source("pubmed", pubmed_fetcher)
        updater.register_source("clinical_trials", trials_fetcher)
        
        # Run update
        job = updater.update_from_source("pubmed")
        
        # Schedule periodic updates
        updater.schedule_updates("pubmed", interval_hours=24)
        updater.start_scheduler()
    """
    
    def __init__(
        self,
        db_manager: Optional[VectorDBManager] = None,
        embedding_pipeline: Optional[EmbeddingPipeline] = None,
        storage_path: Optional[Path] = None
    ):
        self.db_manager = db_manager or VectorDBManager()
        self.embedding_pipeline = embedding_pipeline or EmbeddingPipeline()
        self.content_processor = ContentProcessor()
        
        storage_path = storage_path or Path("models/knowledge_base")
        self.document_tracker = DocumentTracker(storage_path / "tracking")
        self.version_manager = VersionManager(storage_path / "versions")
        
        self._sources: Dict[str, Callable] = {}
        self._jobs: Dict[str, UpdateJob] = {}
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_running = False
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    def register_source(
        self,
        name: str,
        fetcher: Callable[[], List[Dict[str, Any]]]
    ):
        """
        Register a data source.
        
        The fetcher should return a list of dicts with:
        - id: Document ID
        - content: Document content
        - metadata: Optional metadata dict
        """
        self._sources[name] = fetcher
        logger.info(f"Registered source: {name}")
    
    def update_from_source(
        self,
        source_name: str,
        collection: str = "default",
        incremental: bool = True,
        create_version: bool = True
    ) -> UpdateJob:
        """
        Run an update from a registered source.
        
        Args:
            source_name: Name of the registered source
            collection: Target collection
            incremental: Only update new/changed documents
            create_version: Create a new KB version after update
        """
        if source_name not in self._sources:
            raise ValueError(f"Unknown source: {source_name}")
        
        job_id = f"{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = UpdateJob(id=job_id, source=source_name)
        self._jobs[job_id] = job
        
        # Run update in thread
        self._executor.submit(
            self._run_update,
            job, source_name, collection, incremental, create_version
        )
        
        return job
    
    def _run_update(
        self,
        job: UpdateJob,
        source_name: str,
        collection: str,
        incremental: bool,
        create_version: bool
    ):
        """Execute the update job."""
        job.status = UpdateStatus.RUNNING
        job.started_at = datetime.now()
        
        try:
            # Connect to database
            self.db_manager.connect()
            
            # Fetch documents from source
            fetcher = self._sources[source_name]
            raw_documents = fetcher()
            job.documents_processed = len(raw_documents)
            
            logger.info(f"Fetched {len(raw_documents)} documents from {source_name}")
            
            documents_to_add = []
            
            for raw_doc in raw_documents:
                doc_id = raw_doc.get("id", str(hash(raw_doc.get("content", ""))))
                content = raw_doc.get("content", "")
                metadata = raw_doc.get("metadata", {})
                
                # Skip empty documents
                if not content:
                    job.documents_skipped += 1
                    continue
                
                # Check if update needed (incremental mode)
                if incremental and not self.document_tracker.is_new_or_updated(doc_id, content):
                    job.documents_skipped += 1
                    continue
                
                # Process content
                processed = self.content_processor.process(content)
                
                # Skip low quality
                if processed.quality_score < 0.3:
                    job.documents_skipped += 1
                    continue
                
                # Add extracted entities to metadata
                metadata.update({
                    "source": source_name,
                    "diseases": processed.diseases,
                    "drugs": processed.drugs,
                    "quality_score": processed.quality_score,
                    "indexed_at": datetime.now().isoformat()
                })
                
                documents_to_add.append({
                    "id": doc_id,
                    "content": processed.cleaned_text,
                    "metadata": metadata
                })
                
                # Mark as processed
                self.document_tracker.mark_processed(doc_id, content, source_name)
            
            # Embed and insert documents
            if documents_to_add:
                contents = [d["content"] for d in documents_to_add]
                embeddings = self.embedding_pipeline.embed_batch(contents, show_progress=True)
                
                docs = []
                for doc_data, embedding in zip(documents_to_add, embeddings):
                    docs.append(Document(
                        id=doc_data["id"],
                        content=doc_data["content"],
                        embedding=embedding,
                        metadata=doc_data["metadata"],
                        collection=collection
                    ))
                
                self.db_manager.insert(docs)
                job.documents_added = len(docs)
            
            # Create version
            if create_version and job.documents_added > 0:
                total_count = self.db_manager.count(collection)
                self.version_manager.create_version(
                    documents_count=total_count,
                    sources=[source_name],
                    metadata={
                        "job_id": job.id,
                        "added": job.documents_added
                    }
                )
            
            job.status = UpdateStatus.COMPLETED
            logger.info(f"Update completed: {job.documents_added} added, {job.documents_skipped} skipped")
            
        except Exception as e:
            job.status = UpdateStatus.FAILED
            job.error = str(e)
            logger.error(f"Update failed: {e}")
        
        finally:
            job.completed_at = datetime.now()
    
    def schedule_updates(
        self,
        source_name: str,
        interval_hours: int = 24,
        collection: str = "default"
    ):
        """Schedule periodic updates for a source."""
        def scheduled_update():
            logger.info(f"Running scheduled update for {source_name}")
            self.update_from_source(source_name, collection)
        
        schedule.every(interval_hours).hours.do(scheduled_update)
        logger.info(f"Scheduled {source_name} updates every {interval_hours} hours")
    
    def start_scheduler(self):
        """Start the background scheduler."""
        if self._scheduler_running:
            return
        
        self._scheduler_running = True
        
        def run_scheduler():
            while self._scheduler_running:
                schedule.run_pending()
                time.sleep(60)
        
        self._scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self._scheduler_thread.start()
        logger.info("Scheduler started")
    
    def stop_scheduler(self):
        """Stop the background scheduler."""
        self._scheduler_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        schedule.clear()
        logger.info("Scheduler stopped")
    
    def get_job_status(self, job_id: str) -> Optional[UpdateJob]:
        """Get status of an update job."""
        return self._jobs.get(job_id)
    
    def get_all_jobs(self) -> List[UpdateJob]:
        """Get all jobs."""
        return list(self._jobs.values())
    
    def get_current_version(self) -> Optional[KnowledgeVersion]:
        """Get current knowledge base version."""
        return self.version_manager.get_current_version()
    
    def get_version_history(self) -> List[KnowledgeVersion]:
        """Get version history."""
        return self.version_manager.get_version_history()
    
    def get_source_stats(self) -> Dict[str, Any]:
        """Get statistics for all sources."""
        stats = {}
        
        for source_name in self._sources:
            processed_ids = self.document_tracker.get_processed_ids(source_name)
            stats[source_name] = {
                "registered": True,
                "documents_tracked": len(processed_ids)
            }
        
        return stats
    
    def clear_source_tracking(self, source_name: str):
        """Clear tracking data for a source."""
        self.document_tracker.clear(source_name)
        logger.info(f"Cleared tracking for {source_name}")
    
    def add_documents_directly(
        self,
        documents: List[Dict[str, Any]],
        collection: str = "default",
        source: str = "direct"
    ) -> int:
        """
        Add documents directly without a registered source.
        
        Args:
            documents: List of dicts with 'id', 'content', 'metadata'
            collection: Target collection
            source: Source name for tracking
        """
        self.db_manager.connect()
        
        docs_to_add = []
        
        for doc in documents:
            doc_id = doc.get("id", str(hash(doc.get("content", ""))))
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            if not content:
                continue
            
            # Process and embed
            processed = self.content_processor.process(content)
            embedding = self.embedding_pipeline.embed_text(processed.cleaned_text)
            
            metadata.update({
                "source": source,
                "indexed_at": datetime.now().isoformat()
            })
            
            docs_to_add.append(Document(
                id=doc_id,
                content=processed.cleaned_text,
                embedding=embedding,
                metadata=metadata,
                collection=collection
            ))
            
            self.document_tracker.mark_processed(doc_id, content, source)
        
        if docs_to_add:
            self.db_manager.insert(docs_to_add)
        
        return len(docs_to_add)
