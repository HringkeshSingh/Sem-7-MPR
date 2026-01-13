"""
Confidence Scoring System.

Rates generated data based on supporting evidence strength:
- Evidence-based confidence scoring
- Uncertainty estimation for synthetic records
- Multi-dimensional confidence metrics
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level categories."""
    VERY_HIGH = "very_high"      # 0.9-1.0
    HIGH = "high"                 # 0.75-0.9
    MODERATE = "moderate"         # 0.5-0.75
    LOW = "low"                   # 0.25-0.5
    VERY_LOW = "very_low"         # 0-0.25


class UncertaintySource(Enum):
    """Sources of uncertainty in generated data."""
    MODEL_UNCERTAINTY = "model_uncertainty"         # From the generation model itself
    DATA_SCARCITY = "data_scarcity"                # Limited training data
    DISTRIBUTION_SHIFT = "distribution_shift"       # Query differs from training
    CLINICAL_COMPLEXITY = "clinical_complexity"     # Complex medical scenario
    LITERATURE_GAP = "literature_gap"              # Limited literature support


@dataclass
class DimensionalConfidence:
    """Confidence broken down by dimension."""
    clinical_validity: float = 0.0
    statistical_plausibility: float = 0.0
    literature_support: float = 0.0
    temporal_currency: float = 0.0
    internal_consistency: float = 0.0
    
    def overall(self) -> float:
        """Calculate overall confidence."""
        weights = {
            'clinical_validity': 0.30,
            'statistical_plausibility': 0.25,
            'literature_support': 0.20,
            'temporal_currency': 0.10,
            'internal_consistency': 0.15
        }
        
        total = (
            self.clinical_validity * weights['clinical_validity'] +
            self.statistical_plausibility * weights['statistical_plausibility'] +
            self.literature_support * weights['literature_support'] +
            self.temporal_currency * weights['temporal_currency'] +
            self.internal_consistency * weights['internal_consistency']
        )
        
        return total
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "clinical_validity": self.clinical_validity,
            "statistical_plausibility": self.statistical_plausibility,
            "literature_support": self.literature_support,
            "temporal_currency": self.temporal_currency,
            "internal_consistency": self.internal_consistency,
            "overall": self.overall()
        }


@dataclass
class UncertaintyEstimate:
    """Uncertainty estimate for a value or record."""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float  # e.g., 0.95 for 95% CI
    sources: List[UncertaintySource] = field(default_factory=list)
    
    @property
    def range(self) -> float:
        return self.upper_bound - self.lower_bound
    
    @property
    def relative_uncertainty(self) -> float:
        if self.point_estimate == 0:
            return float('inf')
        return self.range / abs(self.point_estimate)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_estimate": self.point_estimate,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence_level": self.confidence_level,
            "range": self.range,
            "relative_uncertainty": self.relative_uncertainty,
            "sources": [s.value for s in self.sources]
        }


@dataclass
class RecordConfidence:
    """Confidence assessment for a single record."""
    record_id: Any
    confidence_score: float
    confidence_level: ConfidenceLevel
    dimensional: DimensionalConfidence
    uncertainties: Dict[str, UncertaintyEstimate] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": str(self.record_id),
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value,
            "dimensional": self.dimensional.to_dict(),
            "uncertainties": {k: v.to_dict() for k, v in self.uncertainties.items()},
            "flags": self.flags
        }


@dataclass
class ConfidenceReport:
    """Complete confidence report for generated data."""
    overall_confidence: float
    confidence_level: ConfidenceLevel
    dimensional_scores: DimensionalConfidence
    record_confidences: Optional[List[RecordConfidence]] = None
    aggregate_uncertainties: Dict[str, UncertaintyEstimate] = field(default_factory=dict)
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "overall_confidence": self.overall_confidence,
            "confidence_level": self.confidence_level.value,
            "dimensional_scores": self.dimensional_scores.to_dict(),
            "aggregate_uncertainties": {k: v.to_dict() for k, v in self.aggregate_uncertainties.items()},
            "evidence_summary": self.evidence_summary,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.record_confidences:
            result["records_assessed"] = len(self.record_confidences)
            result["low_confidence_count"] = sum(
                1 for r in self.record_confidences 
                if r.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]
            )
        
        return result


class ConfidenceScorer:
    """
    Scores confidence in generated healthcare data.
    
    Usage:
        scorer = ConfidenceScorer()
        
        # Score generated data
        report = scorer.score(generated_df)
        
        # Score with validation results
        report = scorer.score_with_validation(
            generated_df,
            clinical_report=clinical_validation,
            literature_report=literature_validation,
            temporal_report=temporal_validation
        )
        
        # Get per-record confidence
        record_scores = scorer.score_records(generated_df)
    """
    
    def __init__(self):
        self._calibration_data: Dict[str, Any] = {}
    
    def score(
        self,
        data: pd.DataFrame,
        query_context: Optional[Dict[str, Any]] = None
    ) -> ConfidenceReport:
        """
        Score confidence in generated data.
        
        Args:
            data: Generated healthcare data
            query_context: Original query context for comparison
        """
        # Calculate dimensional confidence
        dimensional = DimensionalConfidence(
            clinical_validity=self._score_clinical_validity(data),
            statistical_plausibility=self._score_statistical_plausibility(data),
            literature_support=0.6,  # Default without literature validation
            temporal_currency=self._score_temporal_currency(data),
            internal_consistency=self._score_internal_consistency(data)
        )
        
        overall = dimensional.overall()
        level = self._get_confidence_level(overall)
        
        # Calculate aggregate uncertainties
        uncertainties = self._calculate_aggregate_uncertainties(data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(dimensional, uncertainties)
        
        return ConfidenceReport(
            overall_confidence=overall,
            confidence_level=level,
            dimensional_scores=dimensional,
            aggregate_uncertainties=uncertainties,
            recommendations=recommendations
        )
    
    def score_with_validation(
        self,
        data: pd.DataFrame,
        clinical_report: Optional[Any] = None,
        literature_report: Optional[Any] = None,
        temporal_report: Optional[Any] = None
    ) -> ConfidenceReport:
        """
        Score confidence using validation results.
        
        Args:
            data: Generated healthcare data
            clinical_report: ClinicalValidationReport
            literature_report: LiteratureValidationReport
            temporal_report: TemporalValidationReport
        """
        # Extract scores from validation reports
        clinical_score = 0.7  # Default
        if clinical_report:
            clinical_score = getattr(clinical_report, 'overall_score', 0.7)
        
        literature_score = 0.6  # Default
        if literature_report:
            literature_score = getattr(literature_report, 'overall_alignment', 0.6)
        
        temporal_score = 0.8  # Default
        if temporal_report:
            temporal_score = getattr(temporal_report, 'currency_score', 0.8)
        
        # Calculate dimensional confidence
        dimensional = DimensionalConfidence(
            clinical_validity=clinical_score,
            statistical_plausibility=self._score_statistical_plausibility(data),
            literature_support=literature_score,
            temporal_currency=temporal_score,
            internal_consistency=self._score_internal_consistency(data)
        )
        
        overall = dimensional.overall()
        level = self._get_confidence_level(overall)
        
        # Build evidence summary
        evidence = {
            "clinical_validation": clinical_report is not None,
            "literature_validation": literature_report is not None,
            "temporal_validation": temporal_report is not None,
            "sources_count": 0
        }
        
        if literature_report:
            evidence["sources_count"] = getattr(literature_report, 'literature_sources_used', 0)
        
        # Calculate uncertainties
        uncertainties = self._calculate_aggregate_uncertainties(data)
        
        # Adjust uncertainty based on validation
        if clinical_report and hasattr(clinical_report, 'issues_by_severity'):
            critical_issues = clinical_report.issues_by_severity.get('critical', 0)
            if critical_issues > 0:
                for key in uncertainties:
                    uncertainties[key].sources.append(UncertaintySource.CLINICAL_COMPLEXITY)
        
        recommendations = self._generate_recommendations(dimensional, uncertainties)
        
        return ConfidenceReport(
            overall_confidence=overall,
            confidence_level=level,
            dimensional_scores=dimensional,
            aggregate_uncertainties=uncertainties,
            evidence_summary=evidence,
            recommendations=recommendations
        )
    
    def score_records(
        self,
        data: pd.DataFrame,
        top_n_uncertain: Optional[int] = None
    ) -> List[RecordConfidence]:
        """
        Calculate per-record confidence scores.
        
        Args:
            data: Generated healthcare data
            top_n_uncertain: Return only top N most uncertain records
        """
        records = []
        
        for idx, row in data.iterrows():
            # Calculate record-level confidence
            dimensional = self._score_record_dimensions(row, data)
            confidence = dimensional.overall()
            level = self._get_confidence_level(confidence)
            
            # Calculate field uncertainties for this record
            uncertainties = self._calculate_record_uncertainties(row, data)
            
            # Generate flags
            flags = self._generate_record_flags(row, data, confidence)
            
            records.append(RecordConfidence(
                record_id=idx,
                confidence_score=confidence,
                confidence_level=level,
                dimensional=dimensional,
                uncertainties=uncertainties,
                flags=flags
            ))
        
        # Sort by confidence (ascending = least confident first)
        records.sort(key=lambda x: x.confidence_score)
        
        if top_n_uncertain:
            return records[:top_n_uncertain]
        
        return records
    
    def _score_clinical_validity(self, data: pd.DataFrame) -> float:
        """Score clinical validity of the data."""
        scores = []
        
        # Check for out-of-range values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            values = data[col].dropna()
            if len(values) == 0:
                continue
            
            # Check for extreme outliers (beyond 4 std)
            mean, std = values.mean(), values.std()
            if std > 0:
                outliers = ((values < mean - 4*std) | (values > mean + 4*std)).mean()
                scores.append(1.0 - outliers * 2)  # Penalize outliers
        
        # Check for logical consistency
        consistency_score = self._check_logical_consistency(data)
        scores.append(consistency_score)
        
        return np.mean(scores) if scores else 0.7
    
    def _score_statistical_plausibility(self, data: pd.DataFrame) -> float:
        """Score statistical plausibility of distributions."""
        scores = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = data[col].dropna()
            if len(values) < 10:
                continue
            
            # Check for reasonable distribution (not too peaked or flat)
            try:
                # Kurtosis check
                kurtosis = stats.kurtosis(values)
                if -2 <= kurtosis <= 7:  # Reasonable range
                    scores.append(1.0)
                elif -5 <= kurtosis <= 15:
                    scores.append(0.7)
                else:
                    scores.append(0.4)
                
                # Check for mode dominance (synthetic data artifact)
                mode_freq = values.value_counts().iloc[0] / len(values) if len(values) > 0 else 0
                if mode_freq > 0.5 and len(values.unique()) > 5:
                    scores.append(0.5)  # Suspicious mode dominance
                else:
                    scores.append(1.0)
                    
            except Exception:
                scores.append(0.7)
        
        return np.mean(scores) if scores else 0.7
    
    def _score_temporal_currency(self, data: pd.DataFrame) -> float:
        """Score temporal currency of the data."""
        # Simple heuristic - check for date columns in reasonable range
        date_cols = [c for c in data.columns if 'date' in c.lower() or 'year' in c.lower()]
        
        if not date_cols:
            return 0.8  # No dates to check
        
        current_year = datetime.now().year
        scores = []
        
        for col in date_cols:
            try:
                dates = pd.to_datetime(data[col], errors='coerce')
                valid_dates = dates.dropna()
                
                if len(valid_dates) == 0:
                    continue
                
                # Check how recent the data is
                years = valid_dates.dt.year
                recent = (years >= current_year - 5).mean()
                scores.append(recent)
                
                # Check for future dates
                future = (years > current_year).mean()
                if future > 0.01:
                    scores.append(0.5)
                    
            except Exception:
                pass
        
        return np.mean(scores) if scores else 0.8
    
    def _score_internal_consistency(self, data: pd.DataFrame) -> float:
        """Score internal consistency of the data."""
        return self._check_logical_consistency(data)
    
    def _check_logical_consistency(self, data: pd.DataFrame) -> float:
        """Check logical consistency rules."""
        violations = 0
        checks = 0
        
        # ICU LOS <= Hospital LOS
        if 'icu_los_days' in data.columns and 'hospital_los_days' in data.columns:
            checks += 1
            violations += (data['icu_los_days'] > data['hospital_los_days']).sum()
        
        # Age within reasonable range
        if 'age' in data.columns:
            checks += 1
            violations += ((data['age'] < 0) | (data['age'] > 120)).sum()
        
        # Binary columns should be 0 or 1
        binary_cols = [c for c in data.columns if c.startswith('has_') or c.startswith('is_')]
        for col in binary_cols:
            checks += 1
            valid = data[col].isin([0, 1, True, False]).all()
            if not valid:
                violations += 1
        
        if checks == 0:
            return 0.8
        
        return 1.0 - (violations / (len(data) * checks))
    
    def _score_record_dimensions(self, row: pd.Series, data: pd.DataFrame) -> DimensionalConfidence:
        """Calculate dimensional confidence for a single record."""
        # Clinical validity - check if values are within expected ranges
        clinical_score = 0.8
        
        # Check key fields
        if 'age' in row.index:
            age = row['age']
            if pd.notna(age) and 18 <= age <= 100:
                clinical_score += 0.1
            elif pd.notna(age) and (age < 0 or age > 120):
                clinical_score -= 0.3
        
        # Statistical plausibility - how typical is this record
        stat_score = 0.7
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        outlier_count = 0
        for col in numeric_cols:
            if col in row.index and pd.notna(row[col]):
                col_mean = data[col].mean()
                col_std = data[col].std()
                if col_std > 0:
                    z_score = abs((row[col] - col_mean) / col_std)
                    if z_score > 3:
                        outlier_count += 1
        
        if outlier_count == 0:
            stat_score = 0.9
        elif outlier_count <= 2:
            stat_score = 0.7
        else:
            stat_score = 0.4
        
        return DimensionalConfidence(
            clinical_validity=min(clinical_score, 1.0),
            statistical_plausibility=stat_score,
            literature_support=0.6,  # Default for individual records
            temporal_currency=0.8,
            internal_consistency=0.8
        )
    
    def _calculate_aggregate_uncertainties(self, data: pd.DataFrame) -> Dict[str, UncertaintyEstimate]:
        """Calculate aggregate uncertainties for the dataset."""
        uncertainties = {}
        
        # Sample size uncertainty
        n = len(data)
        se_proportion = np.sqrt(0.5 * 0.5 / n) if n > 0 else 1.0  # Worst case SE
        
        uncertainties['sample_representation'] = UncertaintyEstimate(
            point_estimate=1.0,
            lower_bound=max(0, 1.0 - 1.96 * se_proportion),
            upper_bound=min(1.0, 1.0 + 1.96 * se_proportion),
            confidence_level=0.95,
            sources=[UncertaintySource.DATA_SCARCITY] if n < 100 else []
        )
        
        # Distribution uncertainty for key variables
        if 'age' in data.columns:
            ages = data['age'].dropna()
            if len(ages) > 1:
                mean_age = ages.mean()
                se_age = ages.std() / np.sqrt(len(ages))
                
                uncertainties['age_distribution'] = UncertaintyEstimate(
                    point_estimate=mean_age,
                    lower_bound=mean_age - 1.96 * se_age,
                    upper_bound=mean_age + 1.96 * se_age,
                    confidence_level=0.95,
                    sources=[]
                )
        
        return uncertainties
    
    def _calculate_record_uncertainties(
        self,
        row: pd.Series,
        data: pd.DataFrame
    ) -> Dict[str, UncertaintyEstimate]:
        """Calculate uncertainties for a single record."""
        uncertainties = {}
        
        # For numeric fields, estimate uncertainty based on position in distribution
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Limit to first 5 for performance
            if col not in row.index or pd.isna(row[col]):
                continue
            
            col_data = data[col].dropna()
            if len(col_data) < 10:
                continue
            
            value = row[col]
            percentile = stats.percentileofscore(col_data, value) / 100
            
            # Uncertainty higher at extremes
            if percentile < 0.1 or percentile > 0.9:
                uncertainty_factor = 0.3
            elif percentile < 0.25 or percentile > 0.75:
                uncertainty_factor = 0.15
            else:
                uncertainty_factor = 0.1
            
            uncertainties[col] = UncertaintyEstimate(
                point_estimate=value,
                lower_bound=value * (1 - uncertainty_factor),
                upper_bound=value * (1 + uncertainty_factor),
                confidence_level=0.95,
                sources=[UncertaintySource.MODEL_UNCERTAINTY] if percentile < 0.05 or percentile > 0.95 else []
            )
        
        return uncertainties
    
    def _generate_record_flags(
        self,
        row: pd.Series,
        data: pd.DataFrame,
        confidence: float
    ) -> List[str]:
        """Generate flags for a single record."""
        flags = []
        
        if confidence < 0.5:
            flags.append("LOW_CONFIDENCE")
        
        # Check for extreme values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in row.index and pd.notna(row[col]):
                col_data = data[col].dropna()
                if len(col_data) > 10:
                    z_score = (row[col] - col_data.mean()) / col_data.std() if col_data.std() > 0 else 0
                    if abs(z_score) > 3:
                        flags.append(f"EXTREME_{col.upper()}")
        
        # Check for missing critical fields
        critical_fields = ['age', 'gender']
        for field in critical_fields:
            if field in row.index and pd.isna(row[field]):
                flags.append(f"MISSING_{field.upper()}")
        
        return flags[:5]  # Limit flags
    
    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level."""
        if score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.75:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MODERATE
        elif score >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _generate_recommendations(
        self,
        dimensional: DimensionalConfidence,
        uncertainties: Dict[str, UncertaintyEstimate]
    ) -> List[str]:
        """Generate recommendations based on confidence analysis."""
        recommendations = []
        
        if dimensional.clinical_validity < 0.6:
            recommendations.append(
                "Clinical validity is low - review generated values against clinical reference ranges"
            )
        
        if dimensional.statistical_plausibility < 0.6:
            recommendations.append(
                "Statistical plausibility concerns - check for mode collapse or distribution artifacts"
            )
        
        if dimensional.literature_support < 0.5:
            recommendations.append(
                "Limited literature support - consider retrieving more reference documents"
            )
        
        if dimensional.internal_consistency < 0.7:
            recommendations.append(
                "Internal consistency issues - verify logical relationships between fields"
            )
        
        # Check uncertainties
        high_uncertainty = [
            k for k, v in uncertainties.items() 
            if v.relative_uncertainty > 0.5
        ]
        if high_uncertainty:
            recommendations.append(
                f"High uncertainty in: {', '.join(high_uncertainty)}"
            )
        
        return recommendations
