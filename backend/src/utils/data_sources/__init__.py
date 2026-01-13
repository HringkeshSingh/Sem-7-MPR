"""
Multi-source data retrieval system.

This package provides clients for retrieving medical information from multiple sources:
- PubMed (research papers)
- ClinicalTrials.gov (clinical trials)
- WHO (World Health Organization data)
- Medical news and guidelines
"""

from .base_client import BaseDataSourceClient, RetrievedDocument
from .clinical_trials_client import ClinicalTrialsClient
from .who_client import WHOClient
from .medical_news_client import MedicalNewsClient
from .source_aggregator import MultiSourceAggregator

__all__ = [
    'BaseDataSourceClient',
    'RetrievedDocument',
    'ClinicalTrialsClient', 
    'WHOClient',
    'MedicalNewsClient',
    'MultiSourceAggregator'
]
