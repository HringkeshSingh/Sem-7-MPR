"""
Validation Package.

Provides comprehensive validation for generated healthcare data:
- Clinical validation with RAG-augmented verification
- Literature-based statistical validation
- Temporal validation against current medical knowledge
- Confidence scoring with uncertainty estimation
"""

from src.validation.clinical_validator import (
    ClinicalValidator,
    ClinicalValidationReport,
    ValidationResult,
    ValidationIssue,
    ValidationCategory,
    ValidationSeverity
)

from src.validation.literature_validator import (
    LiteratureValidator,
    LiteratureValidationReport,
    StatisticalComparison,
    CoOccurrencePattern,
    PlausibilityFlag
)

from src.validation.temporal_validator import (
    TemporalValidator,
    TemporalValidationReport,
    TemporalIssue,
    TemporalIssueType
)

from src.validation.confidence_scorer import (
    ConfidenceScorer,
    ConfidenceReport,
    RecordConfidence,
    DimensionalConfidence,
    UncertaintyEstimate,
    ConfidenceLevel,
    UncertaintySource
)

__all__ = [
    # Clinical Validator
    'ClinicalValidator',
    'ClinicalValidationReport',
    'ValidationResult',
    'ValidationIssue',
    'ValidationCategory',
    'ValidationSeverity',
    
    # Literature Validator
    'LiteratureValidator',
    'LiteratureValidationReport',
    'StatisticalComparison',
    'CoOccurrencePattern',
    'PlausibilityFlag',
    
    # Temporal Validator
    'TemporalValidator',
    'TemporalValidationReport',
    'TemporalIssue',
    'TemporalIssueType',
    
    # Confidence Scorer
    'ConfidenceScorer',
    'ConfidenceReport',
    'RecordConfidence',
    'DimensionalConfidence',
    'UncertaintyEstimate',
    'ConfidenceLevel',
    'UncertaintySource'
]
