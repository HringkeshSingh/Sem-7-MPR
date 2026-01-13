"""
Global application state management.

Centralizes all global variables and state for the API.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from config.settings import MODELS_DIR, FINAL_DATA_DIR, PROJECT_ROOT
from src.utils.logging_config import get_logger
from src.utils.pubmed_client import PubMedClient
from src.utils.rag_system import HealthcareRAGSystem
from src.utils.enhanced_rag import EnhancedRAGSystem

logger = get_logger(__name__)


class AppState:
    """Centralized application state."""
    
    def __init__(self):
        self.ctgan_model = None
        self.original_dataset: Optional[pd.DataFrame] = None
        self.model_metadata: Optional[Dict[str, Any]] = None
        self.training_columns: Optional[list] = None
        self.pubmed_client: Optional[PubMedClient] = None
        self.rag_system: Optional[HealthcareRAGSystem] = None
        self.enhanced_rag: Optional[EnhancedRAGSystem] = None
        self.rag_generator = None  # RAG-augmented data generator
        self.generation_results: Dict[str, Any] = {}
    
    def load_model_and_data(self):
        """Load the trained CTGAN model and original dataset."""
        try:
            # Load CTGAN model
            model_path = MODELS_DIR / 'ctgan_healthcare_model.pkl'
            if model_path.exists():
                logger.info(f"Loading CTGAN model from {model_path}")
                with open(model_path, 'rb') as f:
                    self.ctgan_model = pickle.load(f)
                logger.info("CTGAN model loaded successfully")
            else:
                logger.warning(f"CTGAN model not found at {model_path}")
            
            # Load model metadata
            metadata_path = MODELS_DIR / 'ctgan_model_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                self.training_columns = self.model_metadata.get('training_columns', [])
                logger.info("Model metadata loaded successfully")
            
            # Load original dataset
            dataset_path = FINAL_DATA_DIR / 'healthcare_dataset_multi_diagnoses.csv'
            if dataset_path.exists():
                logger.info(f"Loading original dataset from {dataset_path}")
                self.original_dataset = pd.read_csv(dataset_path)
                logger.info(f"Original dataset loaded: {len(self.original_dataset)} patients")
            else:
                logger.warning(f"Original dataset not found at {dataset_path}")
            
            # Initialize PubMed client
            self.pubmed_client = PubMedClient(
                api_key=os.getenv("PUBMED_API_KEY"),
                email=os.getenv("PUBMED_EMAIL"),
                tool_name=os.getenv("PUBMED_TOOL_NAME", "healthcare_data_generator")
            )
            logger.info("PubMed client initialized")
            
            # Initialize RAG system
            try:
                self.rag_system = HealthcareRAGSystem(
                    vector_store_path=MODELS_DIR / "rag_vectorstore",
                    similarity_threshold=0.6,
                    top_k=5
                )
                logger.info("RAG system initialized successfully")
                
                # Load existing documentation if available
                docs_path = PROJECT_ROOT / "docs"
                if docs_path.exists():
                    self.rag_system.add_healthcare_documentation(docs_path)
                    logger.info("Added healthcare documentation to RAG system")
            except Exception as e:
                logger.warning(f"Error initializing RAG system: {e}")
                self.rag_system = None
            
            # Initialize Enhanced RAG system
            try:
                self.enhanced_rag = EnhancedRAGSystem(
                    vector_store_path=MODELS_DIR / "enhanced_rag_vectorstore",
                    enable_clinical_trials=True,
                    enable_who=True,
                    enable_medical_news=True,
                    enable_pubmed=True
                )
                logger.info("Enhanced RAG system initialized successfully")
            except Exception as e:
                logger.warning(f"Error initializing Enhanced RAG system: {e}")
                self.enhanced_rag = None
            
            # Initialize RAG data generator
            try:
                from src.api.rag_data_generator import RAGDataGenerator
                self.rag_generator = RAGDataGenerator(
                    ctgan_model=self.ctgan_model,
                    rag_system=self.rag_system,
                    enhanced_rag=self.enhanced_rag,
                    original_dataset=self.original_dataset
                )
                logger.info("RAG data generator initialized successfully")
            except Exception as e:
                logger.warning(f"Error initializing RAG data generator: {e}")
                self.rag_generator = None
                
        except Exception as e:
            logger.error(f"Error loading model and data: {e}")
            raise
    
    def get_enhanced_rag(self) -> EnhancedRAGSystem:
        """Get or initialize the enhanced RAG system."""
        if self.enhanced_rag is None:
            self.enhanced_rag = EnhancedRAGSystem(
                vector_store_path=MODELS_DIR / "enhanced_rag_vectorstore",
                enable_clinical_trials=True,
                enable_who=True,
                enable_medical_news=True,
                enable_pubmed=True
            )
            logger.info("Enhanced RAG system initialized")
        return self.enhanced_rag
    
    @property
    def is_model_loaded(self) -> bool:
        return self.ctgan_model is not None
    
    @property
    def is_dataset_loaded(self) -> bool:
        return self.original_dataset is not None


# Global state instance
app_state = AppState()
