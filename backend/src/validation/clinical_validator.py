"""
Enhanced Clinical Validator.

Provides retrieval-augmented validation of generated healthcare data:
- Cross-checks against retrieved medical literature
- Validates clinical plausibility based on real-world data
- Scores confidence based on supporting evidence
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation checks."""
    CLINICAL_RANGE = "clinical_range"
    DEMOGRAPHIC = "demographic"
    COMORBIDITY = "comorbidity"
    TREATMENT = "treatment"
    TEMPORAL = "temporal"
    STATISTICAL = "statistical"
    LITERATURE = "literature"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in the data."""
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    expected: Optional[str] = None
    actual: Optional[str] = None
    affected_records: int = 0
    evidence: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "affected_records": self.affected_records,
            "evidence": self.evidence
        }


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    score: float  # 0-1 score
    issues: List[ValidationIssue] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    evidence_sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "score": self.score,
            "issues": [i.to_dict() for i in self.issues],
            "details": self.details,
            "evidence_sources": self.evidence_sources
        }


@dataclass
class ClinicalValidationReport:
    """Complete clinical validation report."""
    overall_valid: bool
    overall_score: float
    confidence: float
    total_records: int
    results: List[ValidationResult] = field(default_factory=list)
    issues_by_severity: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_valid": self.overall_valid,
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "total_records": self.total_records,
            "results": [r.to_dict() for r in self.results],
            "issues_by_severity": self.issues_by_severity,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat()
        }


# Clinical reference data
CLINICAL_VALUE_RANGES = {
    'age': {'min': 0, 'max': 120, 'adult_min': 18, 'adult_max': 100},
    'heart_rate': {'min': 30, 'max': 220, 'normal_min': 60, 'normal_max': 100, 'unit': 'bpm'},
    'blood_pressure_systolic': {'min': 50, 'max': 300, 'normal_min': 90, 'normal_max': 140, 'unit': 'mmHg'},
    'blood_pressure_diastolic': {'min': 30, 'max': 180, 'normal_min': 60, 'normal_max': 90, 'unit': 'mmHg'},
    'temperature': {'min': 32.0, 'max': 43.0, 'normal_min': 36.1, 'normal_max': 37.8, 'unit': '°C'},
    'respiratory_rate': {'min': 6, 'max': 60, 'normal_min': 12, 'normal_max': 20, 'unit': '/min'},
    'oxygen_saturation': {'min': 50, 'max': 100, 'normal_min': 95, 'normal_max': 100, 'unit': '%'},
    'glucose': {'min': 20, 'max': 800, 'normal_min': 70, 'normal_max': 140, 'unit': 'mg/dL'},
    'hba1c': {'min': 3.0, 'max': 18.0, 'normal_min': 4.0, 'normal_max': 5.6, 'unit': '%'},
    'creatinine': {'min': 0.1, 'max': 20, 'normal_min': 0.6, 'normal_max': 1.2, 'unit': 'mg/dL'},
    'bmi': {'min': 10, 'max': 70, 'normal_min': 18.5, 'normal_max': 24.9, 'unit': 'kg/m²'},
    'hospital_los_days': {'min': 0, 'max': 365, 'typical_max': 60},
    'icu_los_days': {'min': 0, 'max': 180, 'typical_max': 30}
}

# Expected comorbidity patterns based on medical literature
COMORBIDITY_PATTERNS = {
    'DIABETES': {
        'expected': ['HYPERTENSION', 'CARDIOVASCULAR', 'RENAL', 'OBESITY'],
        'unexpected': [],
        'min_frequency': 0.3  # At least 30% should have comorbidities
    },
    'HYPERTENSION': {
        'expected': ['CARDIOVASCULAR', 'DIABETES', 'RENAL', 'STROKE'],
        'unexpected': [],
        'min_frequency': 0.25
    },
    'CARDIOVASCULAR': {
        'expected': ['HYPERTENSION', 'DIABETES', 'HYPERLIPIDEMIA'],
        'unexpected': [],
        'min_frequency': 0.4
    },
    'RESPIRATORY': {
        'expected': ['CARDIOVASCULAR', 'SMOKING_RELATED'],
        'unexpected': [],
        'min_frequency': 0.2
    },
    'RENAL': {
        'expected': ['DIABETES', 'HYPERTENSION', 'CARDIOVASCULAR'],
        'unexpected': [],
        'min_frequency': 0.5
    },
    'CANCER': {
        'expected': ['ANEMIA', 'MALNUTRITION'],
        'unexpected': [],
        'min_frequency': 0.15
    }
}

# Age-condition relationships
AGE_CONDITION_EXPECTATIONS = {
    'DIABETES_TYPE_1': {'peak_age': (10, 25), 'rare_after': 40},
    'DIABETES_TYPE_2': {'onset_min': 30, 'peak_age': (50, 70)},
    'HYPERTENSION': {'onset_min': 25, 'increases_with_age': True},
    'CARDIOVASCULAR': {'onset_min': 35, 'increases_with_age': True},
    'ALZHEIMER': {'onset_min': 60, 'peak_age': (75, 90)},
    'PEDIATRIC_CONDITIONS': {'max_age': 18}
}


class ClinicalValidator:
    """
    Enhanced clinical validator with retrieval-augmented validation.
    
    Usage:
        validator = ClinicalValidator()
        
        # Validate generated data
        report = validator.validate(generated_df)
        
        # Validate with literature context
        report = validator.validate_with_context(
            generated_df,
            retrieved_documents=pubmed_articles,
            query_context={"conditions": ["DIABETES"], "age_range": (65, 90)}
        )
    """
    
    def __init__(self):
        self._validation_history: List[ClinicalValidationReport] = []
        self._literature_cache: Dict[str, Dict] = {}
    
    def validate(
        self,
        data: pd.DataFrame,
        strict: bool = False
    ) -> ClinicalValidationReport:
        """
        Perform basic clinical validation on generated data.
        
        Args:
            data: Generated healthcare data
            strict: Use stricter validation thresholds
        """
        results = []
        all_issues = []
        
        # Run validation checks
        results.append(self._validate_clinical_ranges(data, strict))
        results.append(self._validate_demographics(data))
        results.append(self._validate_comorbidity_patterns(data))
        results.append(self._validate_clinical_consistency(data))
        results.append(self._validate_data_completeness(data))
        
        # Aggregate issues
        for result in results:
            all_issues.extend(result.issues)
        
        # Calculate overall score
        weights = [1.5, 1.0, 1.2, 1.3, 0.8]  # Weights for each check
        weighted_sum = sum(r.score * w for r, w in zip(results, weights))
        total_weight = sum(weights)
        overall_score = weighted_sum / total_weight
        
        # Count issues by severity
        severity_counts = {}
        for issue in all_issues:
            sev = issue.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        # Determine overall validity
        critical_count = severity_counts.get('critical', 0)
        error_count = severity_counts.get('error', 0)
        overall_valid = critical_count == 0 and error_count <= 2 and overall_score >= 0.7
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, all_issues)
        
        report = ClinicalValidationReport(
            overall_valid=overall_valid,
            overall_score=overall_score,
            confidence=self._calculate_confidence(results),
            total_records=len(data),
            results=results,
            issues_by_severity=severity_counts,
            recommendations=recommendations
        )
        
        self._validation_history.append(report)
        return report
    
    def validate_with_context(
        self,
        data: pd.DataFrame,
        retrieved_documents: Optional[List[Dict]] = None,
        query_context: Optional[Dict[str, Any]] = None,
        strict: bool = False
    ) -> ClinicalValidationReport:
        """
        Validate with retrieval-augmented context from literature.
        
        Args:
            data: Generated healthcare data
            retrieved_documents: PubMed articles or other literature
            query_context: Context from original query (conditions, demographics)
            strict: Use stricter validation
        """
        # Run base validation
        results = []
        all_issues = []
        
        results.append(self._validate_clinical_ranges(data, strict))
        results.append(self._validate_demographics(data))
        results.append(self._validate_comorbidity_patterns(data))
        results.append(self._validate_clinical_consistency(data))
        
        # Add literature-based validation if documents provided
        if retrieved_documents:
            literature_result = self._validate_against_literature(data, retrieved_documents)
            results.append(literature_result)
        
        # Validate against query context
        if query_context:
            context_result = self._validate_query_alignment(data, query_context)
            results.append(context_result)
        
        # Aggregate
        for result in results:
            all_issues.extend(result.issues)
        
        # Calculate scores
        weights = [1.5, 1.0, 1.2, 1.3]
        if retrieved_documents:
            weights.append(1.5)  # Literature validation is important
        if query_context:
            weights.append(1.4)  # Query alignment is important
        
        weighted_sum = sum(r.score * w for r, w in zip(results, weights))
        overall_score = weighted_sum / sum(weights)
        
        severity_counts = {}
        for issue in all_issues:
            sev = issue.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        overall_valid = (
            severity_counts.get('critical', 0) == 0 and
            severity_counts.get('error', 0) <= 2 and
            overall_score >= 0.7
        )
        
        recommendations = self._generate_recommendations(results, all_issues)
        
        # Boost confidence if we have supporting evidence
        base_confidence = self._calculate_confidence(results)
        if retrieved_documents:
            evidence_boost = min(len(retrieved_documents) * 0.02, 0.15)
            confidence = min(base_confidence + evidence_boost, 1.0)
        else:
            confidence = base_confidence
        
        report = ClinicalValidationReport(
            overall_valid=overall_valid,
            overall_score=overall_score,
            confidence=confidence,
            total_records=len(data),
            results=results,
            issues_by_severity=severity_counts,
            recommendations=recommendations
        )
        
        self._validation_history.append(report)
        return report
    
    def _validate_clinical_ranges(
        self,
        data: pd.DataFrame,
        strict: bool = False
    ) -> ValidationResult:
        """Validate clinical values are within realistic ranges."""
        issues = []
        violations = 0
        total_checked = 0
        
        for col in data.columns:
            col_lower = col.lower().replace(' ', '_')
            
            # Find matching range definition
            range_def = None
            for key, ranges in CLINICAL_VALUE_RANGES.items():
                if key in col_lower or col_lower in key:
                    range_def = ranges
                    break
            
            if range_def is None:
                continue
            
            # Check values
            try:
                values = pd.to_numeric(data[col], errors='coerce').dropna()
                if len(values) == 0:
                    continue
                
                total_checked += len(values)
                
                # Check absolute range
                out_of_range = ((values < range_def['min']) | (values > range_def['max']))
                violation_count = out_of_range.sum()
                
                if violation_count > 0:
                    violations += violation_count
                    severity = ValidationSeverity.CRITICAL if violation_count > len(values) * 0.1 else ValidationSeverity.ERROR
                    
                    issues.append(ValidationIssue(
                        category=ValidationCategory.CLINICAL_RANGE,
                        severity=severity,
                        message=f"{col}: {violation_count} values out of valid range",
                        field=col,
                        expected=f"{range_def['min']}-{range_def['max']}",
                        actual=f"min={values.min():.1f}, max={values.max():.1f}",
                        affected_records=int(violation_count)
                    ))
                
                # Check for biologically impossible values in strict mode
                if strict and 'normal_min' in range_def:
                    extreme_low = (values < range_def['min'] * 0.5).sum()
                    extreme_high = (values > range_def['max'] * 1.5).sum()
                    
                    if extreme_low + extreme_high > 0:
                        issues.append(ValidationIssue(
                            category=ValidationCategory.CLINICAL_RANGE,
                            severity=ValidationSeverity.CRITICAL,
                            message=f"{col}: {extreme_low + extreme_high} biologically impossible values",
                            field=col,
                            affected_records=int(extreme_low + extreme_high)
                        ))
                        violations += extreme_low + extreme_high
                        
            except Exception as e:
                logger.warning(f"Error validating {col}: {e}")
        
        if total_checked == 0:
            score = 0.8  # No clinical values to check
        else:
            violation_rate = violations / total_checked
            score = max(0, 1.0 - (violation_rate * 5))  # Penalize violations
        
        return ValidationResult(
            check_name="Clinical Value Ranges",
            passed=score >= 0.8,
            score=score,
            issues=issues,
            details={
                "total_checked": total_checked,
                "violations": violations,
                "violation_rate": violations / max(total_checked, 1)
            }
        )
    
    def _validate_demographics(self, data: pd.DataFrame) -> ValidationResult:
        """Validate demographic data patterns."""
        issues = []
        checks_passed = 0
        total_checks = 0
        
        # Age validation
        if 'age' in data.columns:
            total_checks += 1
            ages = pd.to_numeric(data['age'], errors='coerce').dropna()
            
            if len(ages) > 0:
                # Check for unrealistic ages
                invalid_ages = ((ages < 0) | (ages > 120)).sum()
                if invalid_ages > 0:
                    issues.append(ValidationIssue(
                        category=ValidationCategory.DEMOGRAPHIC,
                        severity=ValidationSeverity.ERROR,
                        message=f"Found {invalid_ages} invalid ages (outside 0-120)",
                        field="age",
                        affected_records=int(invalid_ages)
                    ))
                else:
                    checks_passed += 1
                
                # Check age distribution reasonableness
                total_checks += 1
                mean_age = ages.mean()
                if 30 <= mean_age <= 75:
                    checks_passed += 1
                else:
                    issues.append(ValidationIssue(
                        category=ValidationCategory.DEMOGRAPHIC,
                        severity=ValidationSeverity.WARNING,
                        message=f"Unusual mean age: {mean_age:.1f}",
                        field="age",
                        expected="30-75 for general population",
                        actual=f"{mean_age:.1f}"
                    ))
        
        # Gender validation
        gender_cols = [c for c in data.columns if 'gender' in c.lower() or 'sex' in c.lower()]
        if gender_cols:
            total_checks += 1
            gender_col = gender_cols[0]
            gender_dist = data[gender_col].value_counts(normalize=True)
            
            # Check for reasonable gender distribution
            if len(gender_dist) >= 2:
                min_ratio = gender_dist.min()
                if min_ratio >= 0.2:  # At least 20% of each gender
                    checks_passed += 1
                else:
                    issues.append(ValidationIssue(
                        category=ValidationCategory.DEMOGRAPHIC,
                        severity=ValidationSeverity.WARNING,
                        message=f"Imbalanced gender distribution",
                        field=gender_col,
                        actual=str(gender_dist.to_dict())
                    ))
        
        score = checks_passed / max(total_checks, 1)
        
        return ValidationResult(
            check_name="Demographics Validation",
            passed=score >= 0.7,
            score=score,
            issues=issues,
            details={"checks_passed": checks_passed, "total_checks": total_checks}
        )
    
    def _validate_comorbidity_patterns(self, data: pd.DataFrame) -> ValidationResult:
        """Validate comorbidity patterns match expected medical patterns."""
        issues = []
        pattern_scores = []
        
        # Find diagnosis columns
        diag_cols = [c for c in data.columns if 'diagnos' in c.lower() or 'condition' in c.lower() or c.startswith('has_')]
        
        if not diag_cols:
            return ValidationResult(
                check_name="Comorbidity Patterns",
                passed=True,
                score=0.8,
                issues=[],
                details={"message": "No diagnosis columns found"}
            )
        
        # Extract conditions from data
        condition_counts = {}
        comorbidity_pairs = {}
        
        for idx, row in data.iterrows():
            patient_conditions = set()
            
            for col in diag_cols:
                if col.startswith('has_'):
                    if row.get(col, 0) == 1:
                        condition = col.replace('has_', '').upper()
                        patient_conditions.add(condition)
                elif 'diagnos' in col.lower():
                    val = str(row.get(col, ''))
                    for cond in ['DIABETES', 'HYPERTENSION', 'CARDIOVASCULAR', 'RESPIRATORY', 'RENAL', 'CANCER', 'SEPSIS']:
                        if cond in val.upper():
                            patient_conditions.add(cond)
            
            # Count conditions
            for cond in patient_conditions:
                condition_counts[cond] = condition_counts.get(cond, 0) + 1
            
            # Track comorbidity pairs
            for cond in patient_conditions:
                if cond not in comorbidity_pairs:
                    comorbidity_pairs[cond] = {}
                for other in patient_conditions:
                    if other != cond:
                        comorbidity_pairs[cond][other] = comorbidity_pairs[cond].get(other, 0) + 1
        
        # Validate patterns
        for primary_cond, expected in COMORBIDITY_PATTERNS.items():
            if primary_cond not in condition_counts:
                continue
            
            primary_count = condition_counts[primary_cond]
            if primary_count < 10:  # Not enough data
                continue
            
            # Check if expected comorbidities are present
            actual_comorbidities = comorbidity_pairs.get(primary_cond, {})
            
            has_expected = any(
                cond in actual_comorbidities for cond in expected['expected']
            )
            
            if not has_expected and expected['expected']:
                issues.append(ValidationIssue(
                    category=ValidationCategory.COMORBIDITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"{primary_cond} patients lack expected comorbidities",
                    expected=f"Should have some of: {expected['expected']}",
                    actual=f"Found: {list(actual_comorbidities.keys())}"
                ))
                pattern_scores.append(0.5)
            else:
                pattern_scores.append(1.0)
        
        score = np.mean(pattern_scores) if pattern_scores else 0.8
        
        return ValidationResult(
            check_name="Comorbidity Patterns",
            passed=score >= 0.7,
            score=score,
            issues=issues,
            details={"conditions_analyzed": list(condition_counts.keys())}
        )
    
    def _validate_clinical_consistency(self, data: pd.DataFrame) -> ValidationResult:
        """Validate clinical consistency rules."""
        issues = []
        inconsistencies = 0
        total_records = len(data)
        
        # Check ICU LOS <= Hospital LOS
        if 'icu_los_days' in data.columns and 'hospital_los_days' in data.columns:
            invalid = (data['icu_los_days'] > data['hospital_los_days']).sum()
            if invalid > 0:
                issues.append(ValidationIssue(
                    category=ValidationCategory.CLINICAL_RANGE,
                    severity=ValidationSeverity.ERROR,
                    message="ICU stay longer than hospital stay",
                    affected_records=int(invalid)
                ))
                inconsistencies += invalid
        
        # Check mortality consistency
        if 'mortality' in data.columns and 'discharge_disposition' in data.columns:
            dead_but_discharged = (
                (data['mortality'] == 1) & 
                (data['discharge_disposition'].str.lower().isin(['home', 'rehabilitation', 'snf']))
            ).sum()
            if dead_but_discharged > 0:
                issues.append(ValidationIssue(
                    category=ValidationCategory.CLINICAL_RANGE,
                    severity=ValidationSeverity.CRITICAL,
                    message="Dead patients with living discharge disposition",
                    affected_records=int(dead_but_discharged)
                ))
                inconsistencies += dead_but_discharged
        
        # Diabetes + normal glucose check
        if 'has_diabetes' in data.columns or any('diabetes' in str(c).lower() for c in data.columns):
            glucose_cols = [c for c in data.columns if 'glucose' in c.lower()]
            if glucose_cols:
                diabetic_mask = self._get_condition_mask(data, 'diabetes')
                if diabetic_mask.any():
                    glucose_col = glucose_cols[0]
                    diabetic_glucose = pd.to_numeric(data.loc[diabetic_mask, glucose_col], errors='coerce')
                    low_glucose = (diabetic_glucose < 70).sum()
                    
                    if low_glucose > len(diabetic_glucose) * 0.3:
                        issues.append(ValidationIssue(
                            category=ValidationCategory.CLINICAL_RANGE,
                            severity=ValidationSeverity.WARNING,
                            message="Many diabetic patients with low glucose (<70)",
                            affected_records=int(low_glucose)
                        ))
        
        score = 1.0 - (inconsistencies / max(total_records, 1))
        
        return ValidationResult(
            check_name="Clinical Consistency",
            passed=score >= 0.9,
            score=max(score, 0),
            issues=issues,
            details={"inconsistencies": inconsistencies}
        )
    
    def _validate_data_completeness(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data completeness."""
        issues = []
        
        # Check for required fields
        required_fields = ['age', 'gender']
        missing_required = [f for f in required_fields if f not in data.columns]
        
        if missing_required:
            for field in missing_required:
                issues.append(ValidationIssue(
                    category=ValidationCategory.DEMOGRAPHIC,
                    severity=ValidationSeverity.WARNING,
                    message=f"Missing recommended field: {field}",
                    field=field
                ))
        
        # Check null rates
        null_rates = data.isnull().mean()
        high_null_cols = null_rates[null_rates > 0.5].index.tolist()
        
        for col in high_null_cols:
            issues.append(ValidationIssue(
                category=ValidationCategory.DEMOGRAPHIC,
                severity=ValidationSeverity.WARNING,
                message=f"High null rate in {col}: {null_rates[col]:.1%}",
                field=col
            ))
        
        completeness = 1.0 - null_rates.mean()
        
        return ValidationResult(
            check_name="Data Completeness",
            passed=completeness >= 0.7,
            score=completeness,
            issues=issues,
            details={
                "overall_completeness": completeness,
                "high_null_columns": high_null_cols
            }
        )
    
    def _validate_against_literature(
        self,
        data: pd.DataFrame,
        documents: List[Dict]
    ) -> ValidationResult:
        """Validate data against retrieved medical literature."""
        issues = []
        evidence_sources = []
        alignment_scores = []
        
        # Extract statistics from documents
        literature_stats = self._extract_literature_statistics(documents)
        
        if not literature_stats:
            return ValidationResult(
                check_name="Literature Alignment",
                passed=True,
                score=0.7,
                issues=[],
                details={"message": "No extractable statistics from literature"}
            )
        
        # Compare prevalence rates
        if 'prevalence' in literature_stats:
            for condition, lit_prevalence in literature_stats['prevalence'].items():
                data_prevalence = self._calculate_prevalence(data, condition)
                
                if data_prevalence is not None:
                    # Check if within 2x of literature value
                    ratio = data_prevalence / max(lit_prevalence, 0.01)
                    
                    if 0.5 <= ratio <= 2.0:
                        alignment_scores.append(1.0)
                    elif 0.25 <= ratio <= 4.0:
                        alignment_scores.append(0.7)
                        issues.append(ValidationIssue(
                            category=ValidationCategory.LITERATURE,
                            severity=ValidationSeverity.WARNING,
                            message=f"{condition} prevalence differs from literature",
                            expected=f"{lit_prevalence:.1%}",
                            actual=f"{data_prevalence:.1%}",
                            evidence=f"Source: {literature_stats.get('sources', ['Literature'])[0]}"
                        ))
                    else:
                        alignment_scores.append(0.3)
                        issues.append(ValidationIssue(
                            category=ValidationCategory.LITERATURE,
                            severity=ValidationSeverity.ERROR,
                            message=f"{condition} prevalence significantly differs from literature",
                            expected=f"{lit_prevalence:.1%}",
                            actual=f"{data_prevalence:.1%}"
                        ))
        
        # Compare age distributions
        if 'age_stats' in literature_stats and 'age' in data.columns:
            data_mean_age = data['age'].mean()
            lit_age = literature_stats['age_stats']
            
            if 'mean' in lit_age:
                diff = abs(data_mean_age - lit_age['mean'])
                if diff <= 10:
                    alignment_scores.append(1.0)
                elif diff <= 20:
                    alignment_scores.append(0.7)
                else:
                    alignment_scores.append(0.4)
                    issues.append(ValidationIssue(
                        category=ValidationCategory.LITERATURE,
                        severity=ValidationSeverity.WARNING,
                        message="Mean age differs significantly from literature",
                        expected=f"{lit_age['mean']:.1f}",
                        actual=f"{data_mean_age:.1f}"
                    ))
        
        score = np.mean(alignment_scores) if alignment_scores else 0.7
        evidence_sources = literature_stats.get('sources', [])
        
        return ValidationResult(
            check_name="Literature Alignment",
            passed=score >= 0.6,
            score=score,
            issues=issues,
            evidence_sources=evidence_sources[:5],
            details={
                "checks_performed": len(alignment_scores),
                "literature_sources": len(documents)
            }
        )
    
    def _validate_query_alignment(
        self,
        data: pd.DataFrame,
        query_context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate generated data aligns with original query parameters."""
        issues = []
        alignment_checks = []
        
        # Check condition alignment
        if 'conditions' in query_context:
            expected_conditions = query_context['conditions']
            
            for condition in expected_conditions:
                prevalence = self._calculate_prevalence(data, condition)
                
                if prevalence is not None:
                    if prevalence >= 0.5:  # At least 50% should have the condition
                        alignment_checks.append(1.0)
                    elif prevalence >= 0.2:
                        alignment_checks.append(0.6)
                        issues.append(ValidationIssue(
                            category=ValidationCategory.STATISTICAL,
                            severity=ValidationSeverity.WARNING,
                            message=f"Low prevalence of requested condition: {condition}",
                            expected="≥50%",
                            actual=f"{prevalence:.1%}"
                        ))
                    else:
                        alignment_checks.append(0.2)
                        issues.append(ValidationIssue(
                            category=ValidationCategory.STATISTICAL,
                            severity=ValidationSeverity.ERROR,
                            message=f"Very low prevalence of requested condition: {condition}",
                            expected="≥50%",
                            actual=f"{prevalence:.1%}"
                        ))
        
        # Check age range alignment
        if 'age_range' in query_context and 'age' in data.columns:
            min_age, max_age = query_context['age_range']
            ages = data['age']
            
            in_range = ((ages >= min_age) & (ages <= max_age)).mean()
            
            if in_range >= 0.8:
                alignment_checks.append(1.0)
            elif in_range >= 0.5:
                alignment_checks.append(0.7)
                issues.append(ValidationIssue(
                    category=ValidationCategory.DEMOGRAPHIC,
                    severity=ValidationSeverity.WARNING,
                    message="Some patients outside requested age range",
                    expected=f"{min_age}-{max_age}",
                    actual=f"{in_range:.1%} in range"
                ))
            else:
                alignment_checks.append(0.3)
                issues.append(ValidationIssue(
                    category=ValidationCategory.DEMOGRAPHIC,
                    severity=ValidationSeverity.ERROR,
                    message="Many patients outside requested age range",
                    expected=f"{min_age}-{max_age}",
                    actual=f"{in_range:.1%} in range"
                ))
        
        # Check sample size alignment
        if 'sample_size' in query_context:
            expected_size = query_context['sample_size']
            actual_size = len(data)
            
            ratio = actual_size / expected_size
            if 0.9 <= ratio <= 1.1:
                alignment_checks.append(1.0)
            elif 0.7 <= ratio <= 1.3:
                alignment_checks.append(0.8)
            else:
                alignment_checks.append(0.5)
                issues.append(ValidationIssue(
                    category=ValidationCategory.STATISTICAL,
                    severity=ValidationSeverity.WARNING,
                    message="Sample size differs from request",
                    expected=str(expected_size),
                    actual=str(actual_size)
                ))
        
        score = np.mean(alignment_checks) if alignment_checks else 0.8
        
        return ValidationResult(
            check_name="Query Alignment",
            passed=score >= 0.7,
            score=score,
            issues=issues,
            details={"query_context": query_context}
        )
    
    def _extract_literature_statistics(self, documents: List[Dict]) -> Dict[str, Any]:
        """Extract statistical information from retrieved documents."""
        stats = {
            'prevalence': {},
            'age_stats': {},
            'sources': []
        }
        
        for doc in documents:
            content = doc.get('content', '') or doc.get('abstract', '')
            source = doc.get('source', 'Unknown')
            stats['sources'].append(source)
            
            # Extract prevalence percentages
            prevalence_pattern = r'(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(?:patients?\s+)?(?:with\s+|had\s+)?(\w+)'
            matches = re.findall(prevalence_pattern, content, re.IGNORECASE)
            
            for pct, condition in matches:
                try:
                    value = float(pct) / 100
                    if 0 < value < 1:
                        condition_upper = condition.upper()
                        for cond in ['DIABETES', 'HYPERTENSION', 'CARDIOVASCULAR', 'RESPIRATORY']:
                            if cond in condition_upper or condition_upper in cond:
                                if cond not in stats['prevalence']:
                                    stats['prevalence'][cond] = []
                                stats['prevalence'][cond].append(value)
                except ValueError:
                    pass
            
            # Extract age statistics
            age_pattern = r'(?:mean|average)\s+age\s+(?:was\s+)?(\d+(?:\.\d+)?)'
            age_matches = re.findall(age_pattern, content, re.IGNORECASE)
            
            for age in age_matches:
                try:
                    stats['age_stats'].setdefault('mean_values', []).append(float(age))
                except ValueError:
                    pass
        
        # Average multiple values
        for cond, values in stats['prevalence'].items():
            stats['prevalence'][cond] = np.mean(values) if values else None
        
        if stats['age_stats'].get('mean_values'):
            stats['age_stats']['mean'] = np.mean(stats['age_stats']['mean_values'])
        
        return stats
    
    def _calculate_prevalence(self, data: pd.DataFrame, condition: str) -> Optional[float]:
        """Calculate prevalence of a condition in the data."""
        condition_upper = condition.upper()
        
        # Check binary columns
        binary_col = f'has_{condition.lower()}'
        if binary_col in data.columns:
            return (data[binary_col] == 1).mean()
        
        # Check diagnosis columns
        diag_cols = [c for c in data.columns if 'diagnos' in c.lower()]
        for col in diag_cols:
            matches = data[col].astype(str).str.upper().str.contains(condition_upper, na=False)
            if matches.any():
                return matches.mean()
        
        return None
    
    def _get_condition_mask(self, data: pd.DataFrame, condition: str) -> pd.Series:
        """Get boolean mask for patients with a condition."""
        condition_lower = condition.lower()
        
        # Check binary columns
        binary_col = f'has_{condition_lower}'
        if binary_col in data.columns:
            return data[binary_col] == 1
        
        # Check diagnosis columns
        diag_cols = [c for c in data.columns if 'diagnos' in c.lower()]
        for col in diag_cols:
            return data[col].astype(str).str.lower().str.contains(condition_lower, na=False)
        
        return pd.Series([False] * len(data))
    
    def _calculate_confidence(self, results: List[ValidationResult]) -> float:
        """Calculate overall confidence in the validation."""
        if not results:
            return 0.5
        
        # Weight by number of checks and issues
        total_score = sum(r.score for r in results)
        avg_score = total_score / len(results)
        
        # Reduce confidence for critical issues
        critical_issues = sum(
            1 for r in results for i in r.issues 
            if i.severity == ValidationSeverity.CRITICAL
        )
        
        confidence = avg_score * (0.8 ** critical_issues)
        return max(min(confidence, 1.0), 0.0)
    
    def _generate_recommendations(
        self,
        results: List[ValidationResult],
        issues: List[ValidationIssue]
    ) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Group issues by category
        category_counts = {}
        for issue in issues:
            cat = issue.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Generate recommendations
        if category_counts.get('clinical_range', 0) > 3:
            recommendations.append(
                "Review clinical value generation parameters - multiple out-of-range values detected"
            )
        
        if category_counts.get('comorbidity', 0) > 0:
            recommendations.append(
                "Consider adding comorbidity correlations to the generation model"
            )
        
        if category_counts.get('demographic', 0) > 0:
            recommendations.append(
                "Check demographic distribution parameters for balance"
            )
        
        if category_counts.get('literature', 0) > 2:
            recommendations.append(
                "Generated data significantly differs from published literature - verify source data"
            )
        
        # Check overall scores
        low_scores = [r for r in results if r.score < 0.6]
        if low_scores:
            recommendations.append(
                f"Focus improvement on: {', '.join(r.check_name for r in low_scores)}"
            )
        
        return recommendations
    
    def get_validation_history(self) -> List[ClinicalValidationReport]:
        """Get validation history."""
        return self._validation_history
