"""
Cross-validation system for validating generated data against retrieved sources.

Provides statistical alignment, clinical plausibility, and consistency checks
between generated synthetic data and real-world medical literature.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import re

from src.utils.data_sources.base_client import RetrievedDocument

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    passed: bool
    confidence: float
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class CrossValidationReport:
    """Complete cross-validation report."""
    overall_score: float
    checks_passed: int
    total_checks: int
    results: List[ValidationResult]
    warnings: List[str]
    recommendations: List[str]
    sources_used: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'overall_score': self.overall_score,
            'checks_passed': self.checks_passed,
            'total_checks': self.total_checks,
            'results': [
                {
                    'check_name': r.check_name,
                    'passed': r.passed,
                    'confidence': r.confidence,
                    'message': r.message,
                    'details': r.details
                }
                for r in self.results
            ],
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'sources_used': self.sources_used,
            'timestamp': self.timestamp.isoformat()
        }


class CrossValidator:
    """
    Validates generated data against retrieved medical sources.
    
    Performs multiple validation checks:
    1. Condition prevalence alignment
    2. Age distribution validity
    3. Clinical value ranges
    4. Comorbidity patterns
    5. Treatment/medication consistency
    """
    
    # Reference ranges for clinical values
    CLINICAL_RANGES = {
        'heart_rate': {'min': 40, 'max': 200, 'normal_min': 60, 'normal_max': 100},
        'blood_pressure_systolic': {'min': 60, 'max': 250, 'normal_min': 90, 'normal_max': 140},
        'blood_pressure_diastolic': {'min': 40, 'max': 150, 'normal_min': 60, 'normal_max': 90},
        'temperature': {'min': 35.0, 'max': 42.0, 'normal_min': 36.1, 'normal_max': 37.8},
        'respiratory_rate': {'min': 8, 'max': 40, 'normal_min': 12, 'normal_max': 20},
        'oxygen_saturation': {'min': 70, 'max': 100, 'normal_min': 95, 'normal_max': 100},
        'glucose': {'min': 30, 'max': 600, 'normal_min': 70, 'normal_max': 140},
        'creatinine': {'min': 0.1, 'max': 15, 'normal_min': 0.6, 'normal_max': 1.2}
    }
    
    # Common condition comorbidities
    EXPECTED_COMORBIDITIES = {
        'DIABETES': ['HYPERTENSION', 'CARDIOVASCULAR', 'RENAL'],
        'HYPERTENSION': ['CARDIOVASCULAR', 'DIABETES', 'RENAL'],
        'CARDIOVASCULAR': ['HYPERTENSION', 'DIABETES'],
        'RESPIRATORY': ['CARDIOVASCULAR'],
        'RENAL': ['DIABETES', 'HYPERTENSION', 'CARDIOVASCULAR'],
        'SEPSIS': ['RESPIRATORY', 'RENAL']
    }
    
    def __init__(self):
        """Initialize the cross-validator."""
        self.validation_history = []
    
    def validate(
        self,
        generated_data: Dict[str, Any],
        reference_documents: List[RetrievedDocument],
        query_context: Optional[Dict[str, Any]] = None
    ) -> CrossValidationReport:
        """
        Perform comprehensive cross-validation.
        
        Args:
            generated_data: Generated synthetic data
            reference_documents: Retrieved reference documents
            query_context: Original query context/parameters
            
        Returns:
            CrossValidationReport with all validation results
        """
        results = []
        warnings = []
        recommendations = []
        
        logger.info(f"Starting cross-validation with {len(reference_documents)} reference documents")
        
        # Extract reference information from documents
        reference_info = self._extract_reference_info(reference_documents)
        
        # 1. Validate condition prevalence
        condition_result = self._validate_conditions(generated_data, reference_info)
        results.append(condition_result)
        
        # 2. Validate age distribution
        age_result = self._validate_age_distribution(generated_data, reference_info, query_context)
        results.append(age_result)
        
        # 3. Validate clinical value ranges
        clinical_result = self._validate_clinical_values(generated_data)
        results.append(clinical_result)
        
        # 4. Validate comorbidity patterns
        comorbidity_result = self._validate_comorbidities(generated_data)
        results.append(comorbidity_result)
        
        # 5. Validate source coverage
        coverage_result = self._validate_source_coverage(reference_documents)
        results.append(coverage_result)
        
        # 6. Validate data consistency
        consistency_result = self._validate_data_consistency(generated_data)
        results.append(consistency_result)
        
        # Calculate overall score
        passed_checks = sum(1 for r in results if r.passed)
        total_checks = len(results)
        
        # Weighted average of confidence scores
        if results:
            overall_score = sum(r.confidence for r in results) / len(results)
        else:
            overall_score = 0.0
        
        # Generate warnings and recommendations
        for result in results:
            if not result.passed:
                warnings.append(f"{result.check_name}: {result.message}")
                
        if overall_score < 0.5:
            recommendations.append("Consider adding more reference documents to improve validation")
        if not reference_documents:
            recommendations.append("No reference documents available - results may not be validated")
        if passed_checks < total_checks * 0.7:
            recommendations.append("Review generated data for potential inconsistencies")
        
        report = CrossValidationReport(
            overall_score=overall_score,
            checks_passed=passed_checks,
            total_checks=total_checks,
            results=results,
            warnings=warnings,
            recommendations=recommendations,
            sources_used=len(reference_documents),
            timestamp=datetime.now()
        )
        
        self.validation_history.append(report)
        logger.info(f"Cross-validation complete: {passed_checks}/{total_checks} checks passed, score: {overall_score:.2f}")
        
        return report
    
    def _extract_reference_info(self, documents: List[RetrievedDocument]) -> Dict[str, Any]:
        """Extract structured information from reference documents."""
        info = {
            'conditions': [],
            'age_groups': [],
            'clinical_mentions': {},
            'treatment_mentions': [],
            'source_types': [],
            'total_content_length': 0
        }
        
        condition_patterns = {
            'DIABETES': r'\b(diabetes|diabetic|glucose|insulin|hba1c)\b',
            'HYPERTENSION': r'\b(hypertension|hypertensive|blood pressure|bp)\b',
            'CARDIOVASCULAR': r'\b(cardiovascular|heart|cardiac|coronary|myocardial)\b',
            'RESPIRATORY': r'\b(respiratory|pulmonary|lung|copd|asthma)\b',
            'RENAL': r'\b(renal|kidney|nephro|dialysis|creatinine)\b',
            'SEPSIS': r'\b(sepsis|septic|infection|bacteremia)\b',
            'NEUROLOGICAL': r'\b(neurological|stroke|brain|cerebral)\b',
            'CANCER': r'\b(cancer|tumor|oncolog|malignant|carcinoma)\b'
        }
        
        age_patterns = {
            'elderly': r'\b(elderly|geriatric|aged|older adult|65\+|over 65)\b',
            'adult': r'\b(adult|middle.aged|working.age)\b',
            'pediatric': r'\b(pediatric|child|adolescent|infant|neonatal)\b'
        }
        
        for doc in documents:
            content_lower = doc.content.lower()
            info['total_content_length'] += len(doc.content)
            
            # Extract conditions
            for condition, pattern in condition_patterns.items():
                if re.search(pattern, content_lower, re.IGNORECASE):
                    info['conditions'].append(condition)
            
            # Extract age groups
            for age_group, pattern in age_patterns.items():
                if re.search(pattern, content_lower, re.IGNORECASE):
                    info['age_groups'].append(age_group)
            
            # Track source types
            info['source_types'].append(doc.source)
        
        # Remove duplicates
        info['conditions'] = list(set(info['conditions']))
        info['age_groups'] = list(set(info['age_groups']))
        
        return info
    
    def _validate_conditions(
        self,
        generated_data: Dict[str, Any],
        reference_info: Dict[str, Any]
    ) -> ValidationResult:
        """Validate that generated conditions are supported by references."""
        generated_conditions = []
        
        # Extract conditions from generated data
        if isinstance(generated_data, dict):
            if 'diagnoses' in generated_data:
                generated_conditions = generated_data['diagnoses']
            elif 'data' in generated_data and isinstance(generated_data['data'], list):
                for record in generated_data['data'][:50]:  # Sample first 50
                    if 'diagnosis' in record:
                        generated_conditions.append(record['diagnosis'])
        
        reference_conditions = reference_info.get('conditions', [])
        
        if not generated_conditions and not reference_conditions:
            return ValidationResult(
                check_name="Condition Prevalence",
                passed=True,
                confidence=0.5,
                message="No conditions to validate"
            )
        
        if not reference_conditions:
            return ValidationResult(
                check_name="Condition Prevalence",
                passed=False,
                confidence=0.3,
                message="No reference conditions found in documents"
            )
        
        # Calculate overlap
        gen_set = set(c.upper() if isinstance(c, str) else str(c).upper() for c in generated_conditions)
        ref_set = set(reference_conditions)
        
        if not gen_set:
            return ValidationResult(
                check_name="Condition Prevalence",
                passed=True,
                confidence=0.6,
                message="Reference conditions found but no specific conditions in generated data"
            )
        
        overlap = len(gen_set & ref_set)
        total = len(gen_set | ref_set)
        match_ratio = overlap / max(total, 1)
        
        return ValidationResult(
            check_name="Condition Prevalence",
            passed=match_ratio >= 0.3,
            confidence=match_ratio,
            message=f"Condition alignment: {match_ratio:.0%} ({overlap}/{len(gen_set)} conditions supported)",
            details={
                'generated': list(gen_set),
                'reference': list(ref_set),
                'overlap': list(gen_set & ref_set)
            }
        )
    
    def _validate_age_distribution(
        self,
        generated_data: Dict[str, Any],
        reference_info: Dict[str, Any],
        query_context: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """Validate age distribution against references and query."""
        ages = []
        
        # Extract ages from generated data
        if isinstance(generated_data, dict):
            if 'data' in generated_data and isinstance(generated_data['data'], list):
                for record in generated_data['data'][:100]:
                    if 'age' in record:
                        try:
                            ages.append(float(record['age']))
                        except (ValueError, TypeError):
                            pass
        
        if not ages:
            return ValidationResult(
                check_name="Age Distribution",
                passed=True,
                confidence=0.5,
                message="No age data to validate"
            )
        
        avg_age = sum(ages) / len(ages)
        min_age = min(ages)
        max_age = max(ages)
        
        # Check against query context
        if query_context and query_context.get('age_range'):
            target_min, target_max = query_context['age_range']
            ages_in_range = sum(1 for a in ages if target_min <= a <= target_max)
            compliance_rate = ages_in_range / len(ages)
            
            return ValidationResult(
                check_name="Age Distribution",
                passed=compliance_rate >= 0.8,
                confidence=compliance_rate,
                message=f"Age compliance: {compliance_rate:.0%} within requested range [{target_min}-{target_max}]",
                details={
                    'average_age': avg_age,
                    'range': [min_age, max_age],
                    'target_range': [target_min, target_max]
                }
            )
        
        # Check against reference age groups
        ref_age_groups = reference_info.get('age_groups', [])
        
        if 'elderly' in ref_age_groups and avg_age >= 60:
            return ValidationResult(
                check_name="Age Distribution",
                passed=True,
                confidence=0.8,
                message=f"Age distribution (avg: {avg_age:.1f}) aligns with elderly focus in literature"
            )
        elif 'pediatric' in ref_age_groups and avg_age <= 18:
            return ValidationResult(
                check_name="Age Distribution",
                passed=True,
                confidence=0.8,
                message=f"Age distribution (avg: {avg_age:.1f}) aligns with pediatric focus in literature"
            )
        
        return ValidationResult(
            check_name="Age Distribution",
            passed=True,
            confidence=0.6,
            message=f"Age distribution: avg={avg_age:.1f}, range=[{min_age:.0f}-{max_age:.0f}]",
            details={'average': avg_age, 'min': min_age, 'max': max_age}
        )
    
    def _validate_clinical_values(self, generated_data: Dict[str, Any]) -> ValidationResult:
        """Validate clinical values are within realistic ranges."""
        violations = []
        checked = 0
        
        if isinstance(generated_data, dict) and 'data' in generated_data:
            records = generated_data['data'][:100] if isinstance(generated_data['data'], list) else []
            
            for record in records:
                if not isinstance(record, dict):
                    continue
                    
                for field, ranges in self.CLINICAL_RANGES.items():
                    # Check various field name patterns
                    value = None
                    for key in [field, field.replace('_', ' '), field.upper()]:
                        if key in record:
                            try:
                                value = float(record[key])
                                break
                            except (ValueError, TypeError):
                                pass
                    
                    if value is not None:
                        checked += 1
                        if value < ranges['min'] or value > ranges['max']:
                            violations.append(f"{field}: {value} (valid: {ranges['min']}-{ranges['max']})")
        
        if checked == 0:
            return ValidationResult(
                check_name="Clinical Value Ranges",
                passed=True,
                confidence=0.5,
                message="No clinical values found to validate"
            )
        
        violation_rate = len(violations) / checked
        
        return ValidationResult(
            check_name="Clinical Value Ranges",
            passed=violation_rate < 0.05,  # Allow 5% violations
            confidence=1.0 - violation_rate,
            message=f"Clinical values: {len(violations)} violations out of {checked} checked ({violation_rate:.1%})",
            details={'violations': violations[:10]}  # Limit to 10
        )
    
    def _validate_comorbidities(self, generated_data: Dict[str, Any]) -> ValidationResult:
        """Validate that comorbidity patterns are realistic."""
        if not isinstance(generated_data, dict) or 'data' not in generated_data:
            return ValidationResult(
                check_name="Comorbidity Patterns",
                passed=True,
                confidence=0.5,
                message="No patient data to validate comorbidities"
            )
        
        records = generated_data.get('data', [])
        if not isinstance(records, list):
            return ValidationResult(
                check_name="Comorbidity Patterns",
                passed=True,
                confidence=0.5,
                message="No patient records to validate"
            )
        
        valid_patterns = 0
        total_patterns = 0
        
        for record in records[:50]:
            if not isinstance(record, dict):
                continue
                
            conditions = record.get('diagnoses', [])
            if not isinstance(conditions, list) or len(conditions) < 2:
                continue
            
            # Check if comorbidity patterns make sense
            condition_set = set(c.upper() if isinstance(c, str) else str(c).upper() for c in conditions)
            
            for primary in condition_set:
                if primary in self.EXPECTED_COMORBIDITIES:
                    expected = set(self.EXPECTED_COMORBIDITIES[primary])
                    found_expected = condition_set & expected
                    if found_expected:
                        valid_patterns += 1
                    total_patterns += 1
        
        if total_patterns == 0:
            return ValidationResult(
                check_name="Comorbidity Patterns",
                passed=True,
                confidence=0.6,
                message="No comorbidity patterns to validate"
            )
        
        validity_rate = valid_patterns / total_patterns
        
        return ValidationResult(
            check_name="Comorbidity Patterns",
            passed=validity_rate >= 0.3,
            confidence=validity_rate,
            message=f"Comorbidity validity: {validity_rate:.0%} ({valid_patterns}/{total_patterns} patterns valid)"
        )
    
    def _validate_source_coverage(self, documents: List[RetrievedDocument]) -> ValidationResult:
        """Validate that we have adequate source coverage."""
        if not documents:
            return ValidationResult(
                check_name="Source Coverage",
                passed=False,
                confidence=0.0,
                message="No reference sources available"
            )
        
        sources = list(set(doc.source for doc in documents))
        
        # Ideal is multiple sources
        if len(sources) >= 3:
            confidence = 1.0
            message = f"Excellent coverage: {len(sources)} different sources"
        elif len(sources) >= 2:
            confidence = 0.8
            message = f"Good coverage: {len(sources)} sources ({', '.join(sources)})"
        else:
            confidence = 0.5
            message = f"Limited coverage: only {sources[0]} source"
        
        return ValidationResult(
            check_name="Source Coverage",
            passed=len(sources) >= 1,
            confidence=confidence,
            message=message,
            details={'sources': sources, 'document_count': len(documents)}
        )
    
    def _validate_data_consistency(self, generated_data: Dict[str, Any]) -> ValidationResult:
        """Validate internal data consistency."""
        issues = []
        
        if not isinstance(generated_data, dict):
            return ValidationResult(
                check_name="Data Consistency",
                passed=True,
                confidence=0.5,
                message="No structured data to validate"
            )
        
        records = generated_data.get('data', [])
        if not isinstance(records, list):
            return ValidationResult(
                check_name="Data Consistency",
                passed=True,
                confidence=0.5,
                message="No records to validate"
            )
        
        # Check for consistency issues
        for i, record in enumerate(records[:50]):
            if not isinstance(record, dict):
                continue
            
            # Age/date consistency
            age = record.get('age')
            birth_year = record.get('birth_year')
            if age is not None and birth_year is not None:
                try:
                    calculated_age = 2024 - int(birth_year)
                    if abs(int(age) - calculated_age) > 2:
                        issues.append(f"Record {i}: Age/birth_year mismatch")
                except (ValueError, TypeError):
                    pass
            
            # Gender consistency
            gender = record.get('gender', '')
            if isinstance(gender, str) and gender.lower() not in ['male', 'female', 'm', 'f', '']:
                issues.append(f"Record {i}: Invalid gender value")
        
        if not records:
            return ValidationResult(
                check_name="Data Consistency",
                passed=True,
                confidence=0.5,
                message="No records to check for consistency"
            )
        
        issue_rate = len(issues) / min(len(records), 50)
        
        return ValidationResult(
            check_name="Data Consistency",
            passed=issue_rate < 0.1,
            confidence=1.0 - issue_rate,
            message=f"Data consistency: {len(issues)} issues found",
            details={'issues': issues[:10]}
        )
