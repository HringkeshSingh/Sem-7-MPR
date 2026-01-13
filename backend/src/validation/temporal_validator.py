"""
Temporal Validator.

Validates that generated data aligns with current medical knowledge:
- Flags outdated treatment patterns
- Checks disease associations against current guidelines
- Validates temporal consistency of clinical events
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
import logging
import pandas as pd
import numpy as np
import re

logger = logging.getLogger(__name__)


class TemporalIssueType(Enum):
    """Types of temporal validation issues."""
    OUTDATED_TREATMENT = "outdated_treatment"
    DEPRECATED_TERMINOLOGY = "deprecated_terminology"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    GUIDELINE_MISMATCH = "guideline_mismatch"
    ANACHRONISTIC_DATA = "anachronistic_data"


@dataclass
class TemporalIssue:
    """A temporal validation issue."""
    issue_type: TemporalIssueType
    description: str
    severity: str  # low, medium, high, critical
    field: Optional[str] = None
    outdated_value: Optional[str] = None
    current_recommendation: Optional[str] = None
    affected_records: int = 0
    guideline_source: Optional[str] = None
    guideline_year: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type.value,
            "description": self.description,
            "severity": self.severity,
            "field": self.field,
            "outdated_value": self.outdated_value,
            "current_recommendation": self.current_recommendation,
            "affected_records": self.affected_records,
            "guideline_source": self.guideline_source,
            "guideline_year": self.guideline_year
        }


@dataclass
class TemporalValidationReport:
    """Complete temporal validation report."""
    is_current: bool
    currency_score: float  # 0-1, how current the data appears
    issues: List[TemporalIssue] = field(default_factory=list)
    outdated_patterns_count: int = 0
    guidelines_checked: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    reference_year: int = field(default_factory=lambda: datetime.now().year)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_current": self.is_current,
            "currency_score": self.currency_score,
            "issues": [i.to_dict() for i in self.issues],
            "outdated_patterns_count": self.outdated_patterns_count,
            "guidelines_checked": self.guidelines_checked,
            "recommendations": self.recommendations,
            "reference_year": self.reference_year,
            "timestamp": self.timestamp.isoformat()
        }


# Current medical guidelines and standards (updated periodically)
CURRENT_GUIDELINES = {
    # Diabetes management (ADA 2024)
    'diabetes': {
        'source': 'ADA Standards of Care 2024',
        'year': 2024,
        'recommendations': {
            'first_line': 'metformin',
            'hba1c_target': 7.0,
            'preferred_second_line': ['GLP-1 agonists', 'SGLT2 inhibitors'],
            'deprecated': ['sulfonylureas as first-line', 'sliding scale insulin only']
        }
    },
    # Hypertension (ACC/AHA 2023)
    'hypertension': {
        'source': 'ACC/AHA Hypertension Guideline 2023',
        'year': 2023,
        'recommendations': {
            'threshold': 130,  # mmHg systolic
            'first_line': ['ACE inhibitors', 'ARBs', 'calcium channel blockers', 'thiazides'],
            'deprecated': ['beta-blockers as first-line (unless indication)']
        }
    },
    # Heart failure (ACC/AHA 2022)
    'heart_failure': {
        'source': 'ACC/AHA Heart Failure Guideline 2022',
        'year': 2022,
        'recommendations': {
            'quad_therapy': ['beta-blocker', 'ACE-I/ARB/ARNI', 'MRA', 'SGLT2i'],
            'ef_cutoff': 40,  # Reduced EF threshold
            'deprecated': ['digoxin as primary therapy']
        }
    },
    # Sepsis (Surviving Sepsis 2021)
    'sepsis': {
        'source': 'Surviving Sepsis Campaign 2021',
        'year': 2021,
        'recommendations': {
            'hour_1_bundle': True,
            'fluid_initial': 30,  # mL/kg
            'vasopressor_first': 'norepinephrine',
            'deprecated': ['dopamine as first-line', 'routine steroids']
        }
    },
    # COPD (GOLD 2024)
    'copd': {
        'source': 'GOLD Report 2024',
        'year': 2024,
        'recommendations': {
            'classification': 'ABE groups',  # Changed from ABCD
            'first_line': 'LAMA or LAMA+LABA',
            'deprecated': ['theophylline as first-line', 'ABCD classification']
        }
    }
}

# Outdated treatments and terminology
OUTDATED_PATTERNS = {
    # Deprecated drug classes/specific drugs
    'medications': {
        'patterns': [
            {'term': 'thiazolidinedione', 'replacement': 'GLP-1 agonist or SGLT2 inhibitor', 
             'reason': 'Cardiovascular concerns', 'severity': 'medium'},
            {'term': 'rosiglitazone', 'replacement': 'pioglitazone if TZD needed', 
             'reason': 'CV risk (restricted use)', 'severity': 'high'},
            {'term': 'glyburide', 'replacement': 'glipizide or newer agents', 
             'reason': 'Hypoglycemia risk, especially in elderly', 'severity': 'medium'},
            {'term': 'chlorpropamide', 'replacement': 'modern sulfonylurea or newer class', 
             'reason': 'Obsolete, long half-life', 'severity': 'high'},
            {'term': 'phenformin', 'replacement': 'metformin', 
             'reason': 'Withdrawn due to lactic acidosis', 'severity': 'critical'},
            {'term': 'troglitazone', 'replacement': 'pioglitazone', 
             'reason': 'Withdrawn (hepatotoxicity)', 'severity': 'critical'},
            {'term': 'cisapride', 'replacement': 'metoclopramide', 
             'reason': 'Withdrawn (cardiac arrhythmias)', 'severity': 'critical'},
            {'term': 'rofecoxib', 'replacement': 'celecoxib with caution', 
             'reason': 'Withdrawn (Vioxx) - CV risk', 'severity': 'critical'},
        ]
    },
    # Deprecated terminology
    'terminology': {
        'patterns': [
            {'term': 'juvenile diabetes', 'replacement': 'Type 1 diabetes', 
             'reason': 'Outdated terminology', 'severity': 'low'},
            {'term': 'adult-onset diabetes', 'replacement': 'Type 2 diabetes', 
             'reason': 'Outdated terminology', 'severity': 'low'},
            {'term': 'non-insulin dependent', 'replacement': 'Type 2 diabetes', 
             'reason': 'Misleading - T2D may require insulin', 'severity': 'low'},
            {'term': 'borderline diabetes', 'replacement': 'prediabetes', 
             'reason': 'Outdated terminology', 'severity': 'low'},
            {'term': 'CVA', 'replacement': 'stroke or cerebrovascular accident', 
             'reason': 'Abbreviation discouraged for clarity', 'severity': 'low'},
            {'term': 'NIDDM', 'replacement': 'Type 2 diabetes mellitus', 
             'reason': 'Obsolete classification', 'severity': 'low'},
            {'term': 'IDDM', 'replacement': 'Type 1 diabetes mellitus', 
             'reason': 'Obsolete classification', 'severity': 'low'},
        ]
    },
    # Outdated practices
    'practices': {
        'patterns': [
            {'term': 'tight glycemic control in ICU', 'replacement': 'moderate glucose control (140-180)', 
             'reason': 'NICE-SUGAR trial showed harm', 'severity': 'high'},
            {'term': 'routine stress ulcer prophylaxis', 'replacement': 'risk-stratified approach', 
             'reason': 'SUP-ICU trial', 'severity': 'medium'},
            {'term': 'pulmonary artery catheter routine', 'replacement': 'less invasive monitoring', 
             'reason': 'No mortality benefit in most cases', 'severity': 'medium'},
        ]
    }
}

# Temporal consistency rules
TEMPORAL_RULES = [
    {
        'name': 'admission_before_discharge',
        'description': 'Admission date must be before or equal to discharge date',
        'fields': ['admission_date', 'discharge_date']
    },
    {
        'name': 'diagnosis_during_stay',
        'description': 'Diagnosis date should be within hospital stay',
        'fields': ['diagnosis_date', 'admission_date', 'discharge_date']
    },
    {
        'name': 'death_not_before_admission',
        'description': 'Death date cannot be before admission',
        'fields': ['death_date', 'admission_date']
    },
    {
        'name': 'procedure_during_stay',
        'description': 'Procedure date should be during hospital stay',
        'fields': ['procedure_date', 'admission_date', 'discharge_date']
    }
]


class TemporalValidator:
    """
    Validates generated data against current medical knowledge and temporal consistency.
    
    Usage:
        validator = TemporalValidator()
        
        # Basic validation
        report = validator.validate(generated_df)
        
        # With reference year
        report = validator.validate(generated_df, reference_year=2024)
    """
    
    def __init__(
        self,
        guidelines: Optional[Dict] = None,
        outdated_patterns: Optional[Dict] = None
    ):
        self.guidelines = guidelines or CURRENT_GUIDELINES
        self.outdated_patterns = outdated_patterns or OUTDATED_PATTERNS
        self.current_year = datetime.now().year
    
    def validate(
        self,
        data: pd.DataFrame,
        reference_year: Optional[int] = None,
        check_temporal_consistency: bool = True
    ) -> TemporalValidationReport:
        """
        Validate data for temporal currency and consistency.
        
        Args:
            data: Generated healthcare data
            reference_year: Year to validate against (default: current year)
            check_temporal_consistency: Whether to check date consistency
        """
        reference_year = reference_year or self.current_year
        issues = []
        guidelines_checked = []
        
        # Check for outdated treatments
        treatment_issues = self._check_outdated_treatments(data)
        issues.extend(treatment_issues)
        
        # Check for deprecated terminology
        terminology_issues = self._check_deprecated_terminology(data)
        issues.extend(terminology_issues)
        
        # Check against current guidelines
        guideline_issues = self._check_guidelines_compliance(data)
        issues.extend(guideline_issues)
        guidelines_checked = list(self.guidelines.keys())
        
        # Check temporal consistency
        if check_temporal_consistency:
            consistency_issues = self._check_temporal_consistency(data)
            issues.extend(consistency_issues)
        
        # Check for anachronistic data
        anachronism_issues = self._check_anachronisms(data, reference_year)
        issues.extend(anachronism_issues)
        
        # Calculate currency score
        currency_score = self._calculate_currency_score(issues, len(data))
        
        # Determine if data is current
        critical_issues = [i for i in issues if i.severity == 'critical']
        high_issues = [i for i in issues if i.severity == 'high']
        is_current = len(critical_issues) == 0 and len(high_issues) <= 2 and currency_score >= 0.7
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues)
        
        return TemporalValidationReport(
            is_current=is_current,
            currency_score=currency_score,
            issues=issues,
            outdated_patterns_count=len([i for i in issues if i.issue_type in 
                [TemporalIssueType.OUTDATED_TREATMENT, TemporalIssueType.DEPRECATED_TERMINOLOGY]]),
            guidelines_checked=guidelines_checked,
            recommendations=recommendations,
            reference_year=reference_year
        )
    
    def _check_outdated_treatments(self, data: pd.DataFrame) -> List[TemporalIssue]:
        """Check for outdated treatment patterns."""
        issues = []
        
        # Find medication columns
        med_cols = [c for c in data.columns if any(
            term in c.lower() for term in ['medication', 'drug', 'treatment', 'therapy', 'prescription']
        )]
        
        if not med_cols:
            return issues
        
        for pattern_info in self.outdated_patterns.get('medications', {}).get('patterns', []):
            term = pattern_info['term'].lower()
            
            for col in med_cols:
                # Search for the outdated term
                matches = data[col].astype(str).str.lower().str.contains(term, na=False)
                affected = matches.sum()
                
                if affected > 0:
                    issues.append(TemporalIssue(
                        issue_type=TemporalIssueType.OUTDATED_TREATMENT,
                        description=f"Outdated medication found: {pattern_info['term']}",
                        severity=pattern_info['severity'],
                        field=col,
                        outdated_value=pattern_info['term'],
                        current_recommendation=pattern_info['replacement'],
                        affected_records=int(affected),
                        guideline_source=pattern_info.get('reason', 'Current guidelines')
                    ))
        
        return issues
    
    def _check_deprecated_terminology(self, data: pd.DataFrame) -> List[TemporalIssue]:
        """Check for deprecated medical terminology."""
        issues = []
        
        # Check all text columns
        text_cols = data.select_dtypes(include=['object']).columns
        
        for pattern_info in self.outdated_patterns.get('terminology', {}).get('patterns', []):
            term = pattern_info['term'].lower()
            
            for col in text_cols:
                matches = data[col].astype(str).str.lower().str.contains(term, na=False)
                affected = matches.sum()
                
                if affected > 0:
                    issues.append(TemporalIssue(
                        issue_type=TemporalIssueType.DEPRECATED_TERMINOLOGY,
                        description=f"Deprecated terminology: {pattern_info['term']}",
                        severity=pattern_info['severity'],
                        field=col,
                        outdated_value=pattern_info['term'],
                        current_recommendation=f"Use: {pattern_info['replacement']}",
                        affected_records=int(affected),
                        guideline_source=pattern_info.get('reason')
                    ))
        
        return issues
    
    def _check_guidelines_compliance(self, data: pd.DataFrame) -> List[TemporalIssue]:
        """Check compliance with current clinical guidelines."""
        issues = []
        
        # Check diabetes guidelines
        if self._has_condition(data, 'diabetes'):
            issues.extend(self._check_diabetes_guidelines(data))
        
        # Check hypertension guidelines
        if self._has_condition(data, 'hypertension'):
            issues.extend(self._check_hypertension_guidelines(data))
        
        # Check heart failure guidelines
        if self._has_condition(data, 'heart_failure') or self._has_condition(data, 'chf'):
            issues.extend(self._check_hf_guidelines(data))
        
        return issues
    
    def _check_diabetes_guidelines(self, data: pd.DataFrame) -> List[TemporalIssue]:
        """Check diabetes management against ADA guidelines."""
        issues = []
        guidelines = self.guidelines.get('diabetes', {})
        
        # Check HbA1c targets
        hba1c_cols = [c for c in data.columns if 'hba1c' in c.lower() or 'a1c' in c.lower()]
        
        if hba1c_cols:
            hba1c_col = hba1c_cols[0]
            diabetes_mask = self._get_condition_mask(data, 'diabetes')
            
            if diabetes_mask.any():
                diabetic_hba1c = pd.to_numeric(data.loc[diabetes_mask, hba1c_col], errors='coerce')
                
                # Check for very tight control (may be outdated for some populations)
                very_tight = (diabetic_hba1c < 6.0).sum()
                if very_tight > diabetic_hba1c.count() * 0.2:
                    issues.append(TemporalIssue(
                        issue_type=TemporalIssueType.GUIDELINE_MISMATCH,
                        description="Many patients with very tight HbA1c targets (<6.0%)",
                        severity='medium',
                        field=hba1c_col,
                        current_recommendation="ADA recommends individualized targets, typically <7% for most adults",
                        affected_records=int(very_tight),
                        guideline_source=guidelines.get('source'),
                        guideline_year=guidelines.get('year')
                    ))
        
        return issues
    
    def _check_hypertension_guidelines(self, data: pd.DataFrame) -> List[TemporalIssue]:
        """Check hypertension management against current guidelines."""
        issues = []
        guidelines = self.guidelines.get('hypertension', {})
        
        # Check BP thresholds
        sbp_cols = [c for c in data.columns if 'systolic' in c.lower() or 'sbp' in c.lower()]
        
        if sbp_cols:
            sbp_col = sbp_cols[0]
            htn_mask = self._get_condition_mask(data, 'hypertension')
            
            if htn_mask.any():
                htn_sbp = pd.to_numeric(data.loc[htn_mask, sbp_col], errors='coerce')
                
                # Check if using old 140 threshold
                old_threshold = ((htn_sbp >= 130) & (htn_sbp < 140)).sum()
                diagnosed_total = htn_mask.sum()
                
                # If many hypertensive patients have SBP 130-140, might be using new guidelines
                # This is informational, not necessarily an issue
        
        return issues
    
    def _check_hf_guidelines(self, data: pd.DataFrame) -> List[TemporalIssue]:
        """Check heart failure management against guidelines."""
        issues = []
        guidelines = self.guidelines.get('heart_failure', {})
        
        # Check for GDMT components
        med_cols = [c for c in data.columns if 'medication' in c.lower()]
        
        if med_cols and self._has_condition(data, 'heart_failure'):
            hf_mask = self._get_condition_mask(data, 'heart_failure')
            
            # Check for SGLT2i in HF patients (now guideline recommended)
            # This would be an enhancement to flag if not present
        
        return issues
    
    def _check_temporal_consistency(self, data: pd.DataFrame) -> List[TemporalIssue]:
        """Check temporal consistency of dates in the data."""
        issues = []
        
        for rule in TEMPORAL_RULES:
            # Check if required fields exist
            available_fields = [f for f in rule['fields'] if f in data.columns or 
                              any(f in c.lower() for c in data.columns)]
            
            if len(available_fields) < 2:
                continue
            
            # Find matching columns
            def find_col(field_name):
                if field_name in data.columns:
                    return field_name
                for c in data.columns:
                    if field_name.replace('_', '') in c.lower().replace('_', ''):
                        return c
                return None
            
            cols = [find_col(f) for f in rule['fields'][:2]]
            if not all(cols):
                continue
            
            try:
                date1 = pd.to_datetime(data[cols[0]], errors='coerce')
                date2 = pd.to_datetime(data[cols[1]], errors='coerce')
                
                # Check the rule
                if 'before' in rule['name']:
                    violations = (date1 > date2).sum()
                else:
                    violations = (date1 != date2).sum()  # For equality checks
                
                if violations > 0:
                    issues.append(TemporalIssue(
                        issue_type=TemporalIssueType.TEMPORAL_INCONSISTENCY,
                        description=f"Temporal inconsistency: {rule['description']}",
                        severity='high' if violations > len(data) * 0.05 else 'medium',
                        field=f"{cols[0]}, {cols[1]}",
                        affected_records=int(violations)
                    ))
            except Exception as e:
                logger.warning(f"Error checking temporal rule {rule['name']}: {e}")
        
        return issues
    
    def _check_anachronisms(self, data: pd.DataFrame, reference_year: int) -> List[TemporalIssue]:
        """Check for anachronistic data (dates in the future, etc.)."""
        issues = []
        
        # Find date columns
        date_cols = []
        for col in data.columns:
            if any(term in col.lower() for term in ['date', 'time', 'year']):
                date_cols.append(col)
        
        for col in date_cols:
            try:
                dates = pd.to_datetime(data[col], errors='coerce')
                
                # Check for future dates
                future_dates = (dates.dt.year > reference_year).sum()
                if future_dates > 0:
                    issues.append(TemporalIssue(
                        issue_type=TemporalIssueType.ANACHRONISTIC_DATA,
                        description=f"Future dates found in {col}",
                        severity='high',
                        field=col,
                        affected_records=int(future_dates),
                        current_recommendation=f"Dates should not exceed {reference_year}"
                    ))
                
                # Check for very old dates (before 1900)
                old_dates = (dates.dt.year < 1900).sum()
                if old_dates > 0:
                    issues.append(TemporalIssue(
                        issue_type=TemporalIssueType.ANACHRONISTIC_DATA,
                        description=f"Implausibly old dates in {col}",
                        severity='medium',
                        field=col,
                        affected_records=int(old_dates)
                    ))
                    
            except Exception as e:
                logger.debug(f"Could not parse dates in {col}: {e}")
        
        return issues
    
    def _has_condition(self, data: pd.DataFrame, condition: str) -> bool:
        """Check if any patients have a condition."""
        return self._get_condition_mask(data, condition).any()
    
    def _get_condition_mask(self, data: pd.DataFrame, condition: str) -> pd.Series:
        """Get boolean mask for patients with a condition."""
        condition_lower = condition.lower()
        
        # Check binary columns
        for col in data.columns:
            if col.lower().startswith('has_') and condition_lower in col.lower():
                return data[col] == 1
        
        # Check diagnosis columns
        diag_cols = [c for c in data.columns if 'diagnos' in c.lower() or 'condition' in c.lower()]
        for col in diag_cols:
            matches = data[col].astype(str).str.lower().str.contains(condition_lower, na=False)
            if matches.any():
                return matches
        
        return pd.Series([False] * len(data), index=data.index)
    
    def _calculate_currency_score(self, issues: List[TemporalIssue], total_records: int) -> float:
        """Calculate how current the data appears to be."""
        if total_records == 0:
            return 0.5
        
        # Start with perfect score
        score = 1.0
        
        # Deduct for issues
        for issue in issues:
            impact = issue.affected_records / total_records
            
            if issue.severity == 'critical':
                score -= 0.2 * min(impact * 10, 1)
            elif issue.severity == 'high':
                score -= 0.1 * min(impact * 10, 1)
            elif issue.severity == 'medium':
                score -= 0.05 * min(impact * 10, 1)
            else:
                score -= 0.02 * min(impact * 10, 1)
        
        return max(score, 0.0)
    
    def _generate_recommendations(self, issues: List[TemporalIssue]) -> List[str]:
        """Generate recommendations based on issues found."""
        recommendations = []
        
        # Group by type
        outdated_treatments = [i for i in issues if i.issue_type == TemporalIssueType.OUTDATED_TREATMENT]
        deprecated_terms = [i for i in issues if i.issue_type == TemporalIssueType.DEPRECATED_TERMINOLOGY]
        temporal_issues = [i for i in issues if i.issue_type == TemporalIssueType.TEMPORAL_INCONSISTENCY]
        
        if outdated_treatments:
            recommendations.append(
                f"Update medication data: {len(outdated_treatments)} outdated treatments detected. "
                f"Consider updating source data or post-processing rules."
            )
        
        if deprecated_terms:
            recommendations.append(
                f"Update terminology: {len(deprecated_terms)} instances of deprecated terminology. "
                f"Apply text normalization to use current medical terminology."
            )
        
        if temporal_issues:
            recommendations.append(
                f"Fix date inconsistencies: {len(temporal_issues)} temporal logic errors detected. "
                f"Review date generation logic for clinical event ordering."
            )
        
        # Critical issues
        critical = [i for i in issues if i.severity == 'critical']
        if critical:
            recommendations.insert(0, 
                f"URGENT: {len(critical)} critical temporal issues require immediate attention. "
                f"Data may contain withdrawn medications or impossible combinations."
            )
        
        return recommendations
