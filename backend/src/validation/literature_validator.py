"""
Literature-Based Statistical Validator.

Compares generated data distributions against epidemiological literature:
- Distribution comparison with retrieved data
- Co-occurrence pattern validation
- Statistical plausibility assessment
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class StatisticalComparison:
    """Result of a statistical comparison."""
    metric_name: str
    generated_value: float
    literature_value: float
    difference: float
    p_value: Optional[float] = None
    significant: bool = False
    confidence_interval: Optional[Tuple[float, float]] = None
    sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "generated_value": self.generated_value,
            "literature_value": self.literature_value,
            "difference": self.difference,
            "p_value": self.p_value,
            "significant": self.significant,
            "confidence_interval": self.confidence_interval,
            "sources": self.sources
        }


@dataclass
class CoOccurrencePattern:
    """Co-occurrence pattern from literature."""
    condition1: str
    condition2: str
    expected_correlation: float
    observed_correlation: float
    literature_source: str
    is_plausible: bool
    deviation: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition1": self.condition1,
            "condition2": self.condition2,
            "expected_correlation": self.expected_correlation,
            "observed_correlation": self.observed_correlation,
            "literature_source": self.literature_source,
            "is_plausible": self.is_plausible,
            "deviation": self.deviation
        }


@dataclass
class PlausibilityFlag:
    """Flag for implausible combinations or values."""
    description: str
    severity: str  # low, medium, high, critical
    affected_records: int
    evidence: Optional[str] = None
    recommendation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "severity": self.severity,
            "affected_records": self.affected_records,
            "evidence": self.evidence,
            "recommendation": self.recommendation
        }


@dataclass
class LiteratureValidationReport:
    """Complete literature-based validation report."""
    overall_alignment: float
    statistical_comparisons: List[StatisticalComparison] = field(default_factory=list)
    co_occurrence_patterns: List[CoOccurrencePattern] = field(default_factory=list)
    plausibility_flags: List[PlausibilityFlag] = field(default_factory=list)
    literature_sources_used: int = 0
    confidence_score: float = 0.0
    summary: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_alignment": self.overall_alignment,
            "statistical_comparisons": [c.to_dict() for c in self.statistical_comparisons],
            "co_occurrence_patterns": [p.to_dict() for p in self.co_occurrence_patterns],
            "plausibility_flags": [f.to_dict() for f in self.plausibility_flags],
            "literature_sources_used": self.literature_sources_used,
            "confidence_score": self.confidence_score,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat()
        }


# Reference epidemiological data (from major studies)
EPIDEMIOLOGICAL_REFERENCE = {
    # Prevalence rates in hospitalized patients
    'prevalence': {
        'DIABETES': {'rate': 0.25, 'ci': (0.20, 0.30), 'source': 'CDC NHDS 2020'},
        'HYPERTENSION': {'rate': 0.45, 'ci': (0.40, 0.50), 'source': 'AHA Statistics 2021'},
        'CARDIOVASCULAR': {'rate': 0.30, 'ci': (0.25, 0.35), 'source': 'NHANES 2019'},
        'RESPIRATORY': {'rate': 0.15, 'ci': (0.12, 0.18), 'source': 'CDC NCHS 2020'},
        'RENAL': {'rate': 0.12, 'ci': (0.10, 0.15), 'source': 'USRDS 2020'},
        'SEPSIS': {'rate': 0.08, 'ci': (0.06, 0.10), 'source': 'Sepsis-3 Epidemiology'},
        'CANCER': {'rate': 0.10, 'ci': (0.08, 0.12), 'source': 'SEER 2020'}
    },
    # Age distributions for conditions
    'age_distributions': {
        'DIABETES_TYPE2': {'mean': 62, 'std': 15, 'min': 30, 'source': 'UKPDS'},
        'HYPERTENSION': {'mean': 58, 'std': 18, 'source': 'Framingham Heart Study'},
        'CARDIOVASCULAR': {'mean': 65, 'std': 12, 'source': 'AHA Statistics'},
        'STROKE': {'mean': 70, 'std': 14, 'source': 'Global Stroke Statistics'},
        'COPD': {'mean': 68, 'std': 10, 'source': 'GOLD Report'}
    },
    # Expected co-occurrence rates
    'co_occurrence': {
        ('DIABETES', 'HYPERTENSION'): {'rate': 0.65, 'source': 'ACCORD Trial'},
        ('DIABETES', 'CARDIOVASCULAR'): {'rate': 0.40, 'source': 'Framingham Study'},
        ('DIABETES', 'RENAL'): {'rate': 0.35, 'source': 'UKPDS'},
        ('HYPERTENSION', 'CARDIOVASCULAR'): {'rate': 0.55, 'source': 'SPRINT Trial'},
        ('RESPIRATORY', 'CARDIOVASCULAR'): {'rate': 0.30, 'source': 'COPD-CVD Link Study'},
        ('SEPSIS', 'RENAL'): {'rate': 0.45, 'source': 'Sepsis AKI Studies'}
    },
    # Mortality rates
    'mortality': {
        'overall_icu': {'rate': 0.15, 'ci': (0.12, 0.18), 'source': 'APACHE IV'},
        'sepsis': {'rate': 0.25, 'ci': (0.20, 0.30), 'source': 'Surviving Sepsis'},
        'respiratory_failure': {'rate': 0.20, 'ci': (0.15, 0.25), 'source': 'ARDS Network'}
    }
}

# Implausible combinations
IMPLAUSIBLE_COMBINATIONS = [
    {
        'conditions': ['PEDIATRIC', 'ALZHEIMER'],
        'reason': 'Alzheimer\'s extremely rare in pediatric population',
        'severity': 'critical'
    },
    {
        'conditions': ['TYPE1_DIABETES', 'TYPE2_DIABETES'],
        'reason': 'Patient cannot have both Type 1 and Type 2 diabetes',
        'severity': 'critical'
    },
    {
        'conditions': ['PREGNANCY', 'MALE'],
        'reason': 'Biological impossibility',
        'severity': 'critical'
    },
    {
        'age_condition': ('COPD', '<30'),
        'reason': 'COPD very rare under age 30',
        'severity': 'high'
    },
    {
        'age_condition': ('MENOPAUSE', '<40'),
        'reason': 'Menopause typically occurs after 40',
        'severity': 'medium'
    }
]


class LiteratureValidator:
    """
    Validates generated data against epidemiological literature.
    
    Usage:
        validator = LiteratureValidator()
        
        # Validate with reference data
        report = validator.validate(generated_df)
        
        # Validate with custom literature
        report = validator.validate_with_literature(
            generated_df,
            retrieved_documents=pubmed_articles
        )
    """
    
    def __init__(self, reference_data: Optional[Dict] = None):
        self.reference = reference_data or EPIDEMIOLOGICAL_REFERENCE
        self._extracted_literature: Dict[str, Any] = {}
    
    def validate(
        self,
        data: pd.DataFrame,
        conditions_of_interest: Optional[List[str]] = None
    ) -> LiteratureValidationReport:
        """
        Validate generated data against reference epidemiological data.
        
        Args:
            data: Generated healthcare data
            conditions_of_interest: Specific conditions to focus on
        """
        comparisons = []
        patterns = []
        flags = []
        
        # Prevalence comparisons
        prevalence_results = self._compare_prevalence(data, conditions_of_interest)
        comparisons.extend(prevalence_results)
        
        # Age distribution comparisons
        age_results = self._compare_age_distributions(data, conditions_of_interest)
        comparisons.extend(age_results)
        
        # Co-occurrence pattern validation
        cooccurrence_results = self._validate_co_occurrence(data)
        patterns.extend(cooccurrence_results)
        
        # Check for implausible combinations
        implausibility_flags = self._check_implausible_combinations(data)
        flags.extend(implausibility_flags)
        
        # Check mortality rates if available
        mortality_flags = self._validate_mortality_rates(data)
        flags.extend(mortality_flags)
        
        # Calculate overall alignment
        comparison_scores = []
        for comp in comparisons:
            if comp.literature_value > 0:
                ratio = comp.generated_value / comp.literature_value
                score = 1.0 - min(abs(1 - ratio), 1.0)
                comparison_scores.append(score)
        
        pattern_scores = [1.0 if p.is_plausible else 0.5 for p in patterns]
        
        all_scores = comparison_scores + pattern_scores
        overall_alignment = np.mean(all_scores) if all_scores else 0.7
        
        # Reduce alignment for critical flags
        critical_flags = sum(1 for f in flags if f.severity == 'critical')
        overall_alignment *= (0.8 ** critical_flags)
        
        # Calculate confidence
        confidence = self._calculate_confidence(comparisons, patterns, flags)
        
        # Generate summary
        summary = self._generate_summary(comparisons, patterns, flags, overall_alignment)
        
        return LiteratureValidationReport(
            overall_alignment=overall_alignment,
            statistical_comparisons=comparisons,
            co_occurrence_patterns=patterns,
            plausibility_flags=flags,
            literature_sources_used=len(set(c.sources[0] for c in comparisons if c.sources)),
            confidence_score=confidence,
            summary=summary
        )
    
    def validate_with_literature(
        self,
        data: pd.DataFrame,
        retrieved_documents: List[Dict],
        conditions_of_interest: Optional[List[str]] = None
    ) -> LiteratureValidationReport:
        """
        Validate using both reference data and retrieved literature.
        
        Args:
            data: Generated healthcare data
            retrieved_documents: Retrieved medical literature
            conditions_of_interest: Conditions to focus on
        """
        # Extract statistics from documents
        extracted = self._extract_from_literature(retrieved_documents)
        
        # Merge with reference data
        merged_reference = self._merge_references(extracted)
        
        # Store for use in validation
        original_ref = self.reference
        self.reference = merged_reference
        
        # Run validation
        report = self.validate(data, conditions_of_interest)
        
        # Update sources count
        report.literature_sources_used = len(retrieved_documents)
        
        # Restore original reference
        self.reference = original_ref
        
        return report
    
    def _compare_prevalence(
        self,
        data: pd.DataFrame,
        conditions: Optional[List[str]] = None
    ) -> List[StatisticalComparison]:
        """Compare prevalence rates with reference data."""
        comparisons = []
        
        ref_prevalence = self.reference.get('prevalence', {})
        conditions_to_check = conditions or list(ref_prevalence.keys())
        
        for condition in conditions_to_check:
            if condition not in ref_prevalence:
                continue
            
            ref_data = ref_prevalence[condition]
            observed = self._calculate_prevalence(data, condition)
            
            if observed is None:
                continue
            
            expected = ref_data['rate']
            ci = ref_data.get('ci', (expected * 0.8, expected * 1.2))
            
            # Statistical test
            n = len(data)
            se = np.sqrt(expected * (1 - expected) / n) if n > 0 else 0
            z_score = (observed - expected) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) if se > 0 else 1.0
            
            comparisons.append(StatisticalComparison(
                metric_name=f"{condition} prevalence",
                generated_value=observed,
                literature_value=expected,
                difference=observed - expected,
                p_value=p_value,
                significant=p_value < 0.05,
                confidence_interval=ci,
                sources=[ref_data.get('source', 'Reference')]
            ))
        
        return comparisons
    
    def _compare_age_distributions(
        self,
        data: pd.DataFrame,
        conditions: Optional[List[str]] = None
    ) -> List[StatisticalComparison]:
        """Compare age distributions with reference data."""
        comparisons = []
        
        if 'age' not in data.columns:
            return comparisons
        
        ref_ages = self.reference.get('age_distributions', {})
        conditions_to_check = conditions or list(ref_ages.keys())
        
        for condition in conditions_to_check:
            if condition not in ref_ages:
                continue
            
            ref_data = ref_ages[condition]
            
            # Get ages for patients with this condition
            condition_mask = self._get_condition_mask(data, condition)
            
            if not condition_mask.any():
                continue
            
            condition_ages = data.loc[condition_mask, 'age']
            observed_mean = condition_ages.mean()
            expected_mean = ref_data['mean']
            
            # Two-sample t-test approximation
            n = len(condition_ages)
            expected_std = ref_data.get('std', 15)
            observed_std = condition_ages.std()
            
            se = np.sqrt(expected_std**2/n + observed_std**2/n) if n > 1 else 1
            t_stat = (observed_mean - expected_mean) / se if se > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=max(n-1, 1)))
            
            comparisons.append(StatisticalComparison(
                metric_name=f"{condition} mean age",
                generated_value=observed_mean,
                literature_value=expected_mean,
                difference=observed_mean - expected_mean,
                p_value=p_value,
                significant=p_value < 0.05,
                sources=[ref_data.get('source', 'Reference')]
            ))
        
        return comparisons
    
    def _validate_co_occurrence(self, data: pd.DataFrame) -> List[CoOccurrencePattern]:
        """Validate co-occurrence patterns against literature."""
        patterns = []
        
        ref_cooccur = self.reference.get('co_occurrence', {})
        
        for (cond1, cond2), ref_data in ref_cooccur.items():
            mask1 = self._get_condition_mask(data, cond1)
            mask2 = self._get_condition_mask(data, cond2)
            
            if not mask1.any() or not mask2.any():
                continue
            
            # Calculate observed co-occurrence rate
            # Among patients with cond1, what fraction have cond2?
            with_cond1 = mask1.sum()
            with_both = (mask1 & mask2).sum()
            
            if with_cond1 == 0:
                continue
            
            observed_rate = with_both / with_cond1
            expected_rate = ref_data['rate']
            
            # Determine if plausible (within 2x of expected)
            ratio = observed_rate / max(expected_rate, 0.01)
            is_plausible = 0.5 <= ratio <= 2.0
            
            patterns.append(CoOccurrencePattern(
                condition1=cond1,
                condition2=cond2,
                expected_correlation=expected_rate,
                observed_correlation=observed_rate,
                literature_source=ref_data.get('source', 'Reference'),
                is_plausible=is_plausible,
                deviation=abs(observed_rate - expected_rate)
            ))
        
        return patterns
    
    def _check_implausible_combinations(self, data: pd.DataFrame) -> List[PlausibilityFlag]:
        """Check for biologically implausible combinations."""
        flags = []
        
        for check in IMPLAUSIBLE_COMBINATIONS:
            if 'conditions' in check:
                # Check condition combinations
                cond1, cond2 = check['conditions']
                mask1 = self._get_condition_mask(data, cond1)
                mask2 = self._get_condition_mask(data, cond2)
                
                affected = (mask1 & mask2).sum()
                
                if affected > 0:
                    flags.append(PlausibilityFlag(
                        description=f"Implausible combination: {cond1} + {cond2}",
                        severity=check['severity'],
                        affected_records=int(affected),
                        evidence=check['reason'],
                        recommendation=f"Review records with both {cond1} and {cond2}"
                    ))
            
            elif 'age_condition' in check:
                # Check age-condition combinations
                condition, age_constraint = check['age_condition']
                cond_mask = self._get_condition_mask(data, condition)
                
                if not cond_mask.any() or 'age' not in data.columns:
                    continue
                
                # Parse age constraint
                if age_constraint.startswith('<'):
                    threshold = int(age_constraint[1:])
                    age_mask = data['age'] < threshold
                elif age_constraint.startswith('>'):
                    threshold = int(age_constraint[1:])
                    age_mask = data['age'] > threshold
                else:
                    continue
                
                affected = (cond_mask & age_mask).sum()
                
                if affected > 0:
                    flags.append(PlausibilityFlag(
                        description=f"Unusual age for {condition}: age {age_constraint}",
                        severity=check['severity'],
                        affected_records=int(affected),
                        evidence=check['reason']
                    ))
        
        return flags
    
    def _validate_mortality_rates(self, data: pd.DataFrame) -> List[PlausibilityFlag]:
        """Validate mortality rates against expectations."""
        flags = []
        
        mortality_cols = [c for c in data.columns if 'mortal' in c.lower() or 'death' in c.lower() or 'deceased' in c.lower()]
        
        if not mortality_cols:
            return flags
        
        mortality_col = mortality_cols[0]
        overall_mortality = (data[mortality_col] == 1).mean()
        
        ref_mortality = self.reference.get('mortality', {}).get('overall_icu', {})
        expected = ref_mortality.get('rate', 0.15)
        ci = ref_mortality.get('ci', (0.10, 0.20))
        
        if overall_mortality < ci[0] * 0.5:
            flags.append(PlausibilityFlag(
                description="Mortality rate unusually low",
                severity="medium",
                affected_records=len(data),
                evidence=f"Expected ~{expected:.1%}, observed {overall_mortality:.1%}",
                recommendation="Verify mortality coding is correct"
            ))
        elif overall_mortality > ci[1] * 2:
            flags.append(PlausibilityFlag(
                description="Mortality rate unusually high",
                severity="high",
                affected_records=len(data),
                evidence=f"Expected ~{expected:.1%}, observed {overall_mortality:.1%}",
                recommendation="Check for selection bias in generated data"
            ))
        
        return flags
    
    def _extract_from_literature(self, documents: List[Dict]) -> Dict[str, Any]:
        """Extract statistical information from retrieved documents."""
        extracted = {
            'prevalence': defaultdict(list),
            'age_distributions': defaultdict(list),
            'co_occurrence': defaultdict(list),
            'sources': []
        }
        
        for doc in documents:
            content = doc.get('content', '') or doc.get('abstract', '')
            source = doc.get('title', '') or doc.get('source', 'Unknown')
            extracted['sources'].append(source)
            
            # Extract prevalence percentages
            prev_patterns = [
                r'prevalence\s+(?:of\s+)?(\w+)\s+(?:was\s+)?(\d+(?:\.\d+)?)\s*%',
                r'(\d+(?:\.\d+)?)\s*%\s+(?:of\s+)?patients?\s+(?:had|with)\s+(\w+)',
                r'(\w+)\s+(?:was\s+)?(?:present\s+)?in\s+(\d+(?:\.\d+)?)\s*%'
            ]
            
            for pattern in prev_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    try:
                        if pattern.startswith(r'prevalence'):
                            condition, pct = match
                        elif pattern.startswith(r'(\d'):
                            pct, condition = match
                        else:
                            condition, pct = match
                        
                        value = float(pct) / 100
                        if 0 < value < 1:
                            condition_upper = condition.upper()
                            for std_cond in ['DIABETES', 'HYPERTENSION', 'CARDIOVASCULAR', 'RESPIRATORY', 'RENAL', 'SEPSIS', 'CANCER']:
                                if std_cond in condition_upper or condition_upper in std_cond:
                                    extracted['prevalence'][std_cond].append({
                                        'rate': value,
                                        'source': source[:50]
                                    })
                    except (ValueError, IndexError):
                        pass
            
            # Extract age statistics
            age_patterns = [
                r'mean\s+age\s+(?:was\s+)?(\d+(?:\.\d+)?)\s*(?:\±\s*(\d+(?:\.\d+)?))?',
                r'average\s+age\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*years?'
            ]
            
            for pattern in age_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    try:
                        if isinstance(match, tuple):
                            mean_age = float(match[0])
                            std = float(match[1]) if len(match) > 1 and match[1] else 15
                        else:
                            mean_age = float(match)
                            std = 15
                        
                        extracted['age_distributions']['general'].append({
                            'mean': mean_age,
                            'std': std,
                            'source': source[:50]
                        })
                    except (ValueError, IndexError):
                        pass
        
        return dict(extracted)
    
    def _merge_references(self, extracted: Dict) -> Dict[str, Any]:
        """Merge extracted literature with reference data."""
        merged = {
            'prevalence': dict(self.reference.get('prevalence', {})),
            'age_distributions': dict(self.reference.get('age_distributions', {})),
            'co_occurrence': dict(self.reference.get('co_occurrence', {})),
            'mortality': dict(self.reference.get('mortality', {}))
        }
        
        # Merge prevalence (average with existing)
        for condition, values in extracted.get('prevalence', {}).items():
            if not values:
                continue
            
            avg_rate = np.mean([v['rate'] for v in values])
            sources = [v['source'] for v in values]
            
            if condition in merged['prevalence']:
                # Average with existing
                existing = merged['prevalence'][condition]['rate']
                merged['prevalence'][condition] = {
                    'rate': (existing + avg_rate) / 2,
                    'ci': (min(existing, avg_rate) * 0.8, max(existing, avg_rate) * 1.2),
                    'source': f"Combined: {sources[0]}"
                }
            else:
                merged['prevalence'][condition] = {
                    'rate': avg_rate,
                    'ci': (avg_rate * 0.7, avg_rate * 1.3),
                    'source': sources[0]
                }
        
        return merged
    
    def _calculate_prevalence(self, data: pd.DataFrame, condition: str) -> Optional[float]:
        """Calculate prevalence of a condition."""
        condition_mask = self._get_condition_mask(data, condition)
        if condition_mask.any():
            return condition_mask.mean()
        return None
    
    def _get_condition_mask(self, data: pd.DataFrame, condition: str) -> pd.Series:
        """Get boolean mask for patients with a condition."""
        condition_lower = condition.lower().replace('_', '')
        
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
    
    def _calculate_confidence(
        self,
        comparisons: List[StatisticalComparison],
        patterns: List[CoOccurrencePattern],
        flags: List[PlausibilityFlag]
    ) -> float:
        """Calculate confidence in the validation results."""
        # Start with base confidence based on number of checks
        base = min(0.5 + len(comparisons) * 0.05, 0.8)
        
        # Boost for consistent results
        if comparisons:
            consistent = sum(1 for c in comparisons if not c.significant) / len(comparisons)
            base += consistent * 0.15
        
        # Reduce for flags
        flag_penalty = sum(
            0.1 if f.severity == 'critical' else 0.05 if f.severity == 'high' else 0.02
            for f in flags
        )
        
        return max(min(base - flag_penalty, 1.0), 0.0)
    
    def _generate_summary(
        self,
        comparisons: List[StatisticalComparison],
        patterns: List[CoOccurrencePattern],
        flags: List[PlausibilityFlag],
        alignment: float
    ) -> str:
        """Generate human-readable summary."""
        parts = []
        
        if alignment >= 0.8:
            parts.append("Generated data shows strong alignment with epidemiological literature.")
        elif alignment >= 0.6:
            parts.append("Generated data shows moderate alignment with literature, with some deviations.")
        else:
            parts.append("Generated data shows significant deviations from expected epidemiological patterns.")
        
        # Summarize comparisons
        sig_comparisons = [c for c in comparisons if c.significant]
        if sig_comparisons:
            parts.append(f"{len(sig_comparisons)} of {len(comparisons)} statistical comparisons showed significant differences.")
        
        # Summarize patterns
        implausible = [p for p in patterns if not p.is_plausible]
        if implausible:
            parts.append(f"{len(implausible)} co-occurrence patterns deviate from literature expectations.")
        
        # Summarize flags
        critical = [f for f in flags if f.severity == 'critical']
        if critical:
            parts.append(f"CRITICAL: {len(critical)} implausible combinations detected.")
        
        return " ".join(parts)
