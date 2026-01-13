"""
RAG-Augmented Data Generator.

Generates synthetic healthcare data using retrieval-augmented context:
- Retrieves relevant medical literature for each generation request
- Uses retrieved information to guide synthetic data creation
- Implements multi-hop reasoning for complex medical scenarios
- Provides evidence citations with generated data
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
import re
import json

from src.models.ctgan_trainer import (
    RAGAugmentedCTGANTrainer,
    GenerationContext,
    LiteratureConstraint,
    TrainingAugmentation
)

logger = logging.getLogger(__name__)


@dataclass
class EvidenceCitation:
    """Citation for supporting evidence."""
    source: str
    source_type: str  # 'pubmed', 'clinical_trial', 'who', 'literature'
    title: Optional[str] = None
    url: Optional[str] = None
    relevance_score: float = 0.0
    excerpt: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "source_type": self.source_type,
            "title": self.title,
            "url": self.url,
            "relevance_score": self.relevance_score,
            "excerpt": self.excerpt
        }


@dataclass
class ReasoningStep:
    """A step in multi-hop reasoning."""
    step_number: int
    description: str
    query: str
    result_summary: str
    confidence: float
    sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "description": self.description,
            "query": self.query,
            "result_summary": self.result_summary,
            "confidence": self.confidence,
            "sources": self.sources
        }


@dataclass
class GenerationResult:
    """Result of RAG-augmented data generation."""
    data: pd.DataFrame
    num_records: int
    confidence_score: float
    citations: List[EvidenceCitation] = field(default_factory=list)
    reasoning_chain: List[ReasoningStep] = field(default_factory=list)
    applied_constraints: List[Dict] = field(default_factory=list)
    generation_context: Optional[GenerationContext] = None
    generation_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self, include_data: bool = False) -> Dict[str, Any]:
        result = {
            "num_records": self.num_records,
            "confidence_score": self.confidence_score,
            "citations": [c.to_dict() for c in self.citations],
            "reasoning_chain": [r.to_dict() for r in self.reasoning_chain],
            "applied_constraints": self.applied_constraints,
            "generation_context": self.generation_context.to_dict() if self.generation_context else None,
            "generation_time_ms": self.generation_time_ms,
            "timestamp": self.timestamp.isoformat()
        }
        
        if include_data:
            result["data"] = self.data.to_dict(orient='records')
        
        return result


# Medical ontology for entity extraction and expansion
MEDICAL_ONTOLOGY = {
    'diabetes': {
        'icd10': ['E10', 'E11', 'E13'],
        'related': ['insulin resistance', 'hyperglycemia', 'diabetic nephropathy', 'diabetic retinopathy'],
        'risk_factors': ['obesity', 'age', 'family_history', 'sedentary_lifestyle'],
        'complications': ['cardiovascular', 'renal', 'neuropathy', 'retinopathy'],
        'synonyms': ['diabetes mellitus', 'dm', 'sugar diabetes']
    },
    'hypertension': {
        'icd10': ['I10', 'I11', 'I12', 'I13'],
        'related': ['high blood pressure', 'elevated bp', 'resistant hypertension'],
        'risk_factors': ['obesity', 'salt_intake', 'stress', 'age'],
        'complications': ['stroke', 'cardiovascular', 'renal'],
        'synonyms': ['htn', 'high blood pressure', 'elevated blood pressure']
    },
    'cardiovascular': {
        'icd10': ['I20', 'I21', 'I25', 'I50'],
        'related': ['coronary artery disease', 'heart failure', 'myocardial infarction'],
        'risk_factors': ['hypertension', 'diabetes', 'smoking', 'hyperlipidemia'],
        'complications': ['heart_failure', 'arrhythmia', 'sudden_death'],
        'synonyms': ['cvd', 'heart disease', 'cardiac']
    },
    'sepsis': {
        'icd10': ['A41', 'R65.2'],
        'related': ['septic shock', 'bacteremia', 'severe infection'],
        'risk_factors': ['immunosuppression', 'diabetes', 'elderly', 'recent_surgery'],
        'complications': ['organ_failure', 'dic', 'ards'],
        'synonyms': ['septicemia', 'blood poisoning']
    },
    'respiratory': {
        'icd10': ['J44', 'J45', 'J96'],
        'related': ['copd', 'asthma', 'respiratory failure', 'pneumonia'],
        'risk_factors': ['smoking', 'pollution', 'occupational_exposure'],
        'complications': ['respiratory_failure', 'pulmonary_hypertension'],
        'synonyms': ['lung disease', 'pulmonary']
    },
    'renal': {
        'icd10': ['N17', 'N18', 'N19'],
        'related': ['chronic kidney disease', 'acute kidney injury', 'dialysis'],
        'risk_factors': ['diabetes', 'hypertension', 'nephrotoxic_drugs'],
        'complications': ['esrd', 'cardiovascular', 'anemia'],
        'synonyms': ['kidney disease', 'ckd', 'aki', 'nephropathy']
    },
    'neurological': {
        'icd10': ['G00', 'G81', 'G35', 'G40'],
        'related': ['neurological disease', 'neurological disorder', 'neuro disorder', 'neuro condition'],
        'risk_factors': ['stroke', 'seizure', 'trauma'],
        'complications': ['cognitive_impairment', 'motor_impairment'],
        'synonyms': ['neuro', 'neurologic', 'brain disorder', 'neuro condition', 'neurological diseases']
    }
}


class RAGDataGenerator:
    """
    RAG-augmented synthetic healthcare data generator.
    
    Usage:
        generator = RAGDataGenerator(
            ctgan_model=trained_model,
            rag_system=rag_system,
            enhanced_rag=enhanced_rag
        )
        
        # Generate with context retrieval
        result = generator.generate(
            query="Generate 100 elderly diabetic patients with renal complications",
            num_samples=100
        )
        
        # Access generated data and evidence
        print(f"Generated {result.num_records} records")
        print(f"Confidence: {result.confidence_score:.2%}")
        for citation in result.citations:
            print(f"  - {citation.source}: {citation.title}")
    """
    
    def __init__(
        self,
        ctgan_model: Any = None,
        rag_system: Any = None,
        enhanced_rag: Any = None,
        original_dataset: Optional[pd.DataFrame] = None
    ):
        self.ctgan_model = ctgan_model
        self.rag_system = rag_system
        self.enhanced_rag = enhanced_rag
        self.original_dataset = original_dataset
        
        self._trainer = RAGAugmentedCTGANTrainer(
            rag_system=rag_system,
            enhanced_rag=enhanced_rag
        )
        
        self._generation_history: List[GenerationResult] = []
    
    def generate(
        self,
        query: str,
        num_samples: int = 100,
        use_multi_hop: bool = True,
        include_citations: bool = True
    ) -> GenerationResult:
        """
        Generate synthetic data using RAG-augmented context.
        
        Args:
            query: Natural language query describing desired data
            num_samples: Number of records to generate
            use_multi_hop: Use multi-hop reasoning for complex queries
            include_citations: Include supporting evidence citations
        """
        start_time = datetime.now()
        
        # Parse the query to extract entities and requirements
        parsed = self._parse_generation_query(query)
        
        # Multi-hop reasoning for complex scenarios
        reasoning_chain = []
        if use_multi_hop and self._is_complex_query(parsed):
            reasoning_chain = self._multi_hop_reasoning(query, parsed)
        
        # Build generation context
        context = self._build_generation_context(query, parsed, reasoning_chain)
        
        # Collect citations
        citations = []
        if include_citations:
            citations = self._collect_citations(query, context)
        
        # Generate data
        if self.ctgan_model:
            synthetic_df = self._trainer.generate_with_context(
                self.ctgan_model,
                num_samples,
                context,
                post_process=True
            )
        else:
            # Fallback to sampling from original dataset with constraints
            synthetic_df = self._fallback_generate(num_samples, context)
        
        # Calculate confidence
        confidence = self._calculate_generation_confidence(
            synthetic_df, context, reasoning_chain, citations
        )
        
        generation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = GenerationResult(
            data=synthetic_df,
            num_records=len(synthetic_df),
            confidence_score=confidence,
            citations=citations,
            reasoning_chain=reasoning_chain,
            applied_constraints=[c.to_dict() for c in context.constraints],
            generation_context=context,
            generation_time_ms=generation_time
        )
        
        self._generation_history.append(result)
        
        logger.info(
            f"Generated {len(synthetic_df)} records with confidence {confidence:.2%} "
            f"in {generation_time:.1f}ms"
        )
        
        return result
    
    def generate_with_validation(
        self,
        query: str,
        num_samples: int = 100,
        validate: bool = True
    ) -> Tuple[GenerationResult, Optional[Dict]]:
        """
        Generate data and optionally validate against literature.
        
        Args:
            query: Generation query
            num_samples: Number of samples
            validate: Whether to validate generated data
            
        Returns:
            Tuple of (GenerationResult, ValidationReport dict or None)
        """
        result = self.generate(query, num_samples, use_multi_hop=True)
        
        validation_report = None
        
        if validate and self.enhanced_rag:
            try:
                validation = self.enhanced_rag.validate_generated_data(
                    {'data': result.data.to_dict(orient='records')},
                    query=query
                )
                validation_report = validation.to_dict() if hasattr(validation, 'to_dict') else validation
            except Exception as e:
                logger.warning(f"Validation failed: {e}")
        
        return result, validation_report
    
    def _parse_generation_query(self, query: str) -> Dict[str, Any]:
        """Parse query to extract generation requirements."""
        query_lower = query.lower()
        
        parsed = {
            'conditions': [],
            'age_descriptors': [],
            'age_range': None,
            'gender': None,
            'gender_ratio': 1.0,  # 1.0 = hard constraint, <1.0 = soft constraint
            'severity': None,
            'icu_required': False,
            'mortality_context': None,
            'sample_size': None,
            'complications': [],
            'risk_factors': []
        }
        
        # Extract conditions using ontology
        for condition, info in MEDICAL_ONTOLOGY.items():
            all_terms = [condition] + info.get('synonyms', [])
            if any(term in query_lower for term in all_terms):
                parsed['conditions'].append(condition.upper())
        
        # Extract age information
        age_patterns = [
            (r'elderly|geriatric|old|aged', (65, 100)),
            (r'middle.?aged?', (40, 65)),
            (r'young adult', (18, 40)),
            (r'adult', (18, 65)),
            (r'pediatric|child', (0, 18)),
            (r'(\d+)\s*(?:to|-)\s*(\d+)\s*(?:years?|y/?o)?', None),  # Explicit range
            (r'over\s*(\d+)', None),  # Over X years
            (r'under\s*(\d+)', None),  # Under X years
        ]
        
        for pattern, default_range in age_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if default_range:
                    parsed['age_range'] = default_range
                    parsed['age_descriptors'].append(match.group())
                elif 'to' in pattern or '-' in pattern:
                    parsed['age_range'] = (int(match.group(1)), int(match.group(2)))
                elif 'over' in pattern:
                    parsed['age_range'] = (int(match.group(1)), 100)
                elif 'under' in pattern:
                    parsed['age_range'] = (0, int(match.group(1)))
                break
        
        # Extract gender - use word boundaries to avoid "male" matching inside "female"
        # Check female FIRST since "male" is a substring of "female"
        female_pattern = r'\b(female|females|women|woman)\b'
        male_pattern = r'\b(male|males|men|man)\b(?!\s*female)'  # negative lookahead to avoid "male" in "female"
        
        if re.search(female_pattern, query_lower):
            parsed['gender'] = 'female'
        elif re.search(male_pattern, query_lower):
            parsed['gender'] = 'male'
        
        # Check for soft constraints ("mostly", "predominantly", "majority")
        soft_gender_pattern = r'\b(mostly|predominantly|majority|primarily|mainly)\b'
        if re.search(soft_gender_pattern, query_lower):
            parsed['gender_ratio'] = 0.75  # 75% target gender, 25% other
        else:
            parsed['gender_ratio'] = 1.0   # 100% target gender (hard constraint)
        
        # Extract severity
        if any(s in query_lower for s in ['critical', 'severe', 'life-threatening']):
            parsed['severity'] = 'critical'
        elif any(s in query_lower for s in ['serious', 'major']):
            parsed['severity'] = 'high'
        elif any(s in query_lower for s in ['moderate', 'medium']):
            parsed['severity'] = 'medium'
        elif any(s in query_lower for s in ['mild', 'minor']):
            parsed['severity'] = 'low'
        
        # ICU requirement
        if any(term in query_lower for term in ['icu', 'intensive care', 'critical care']):
            parsed['icu_required'] = True
        
        # Sample size
        size_match = re.search(r'(?:generate|create)?\s*(\d+)\s*(?:patients?|records?|samples?)', query_lower)
        if size_match:
            parsed['sample_size'] = int(size_match.group(1))
        
        # Extract complications based on ontology
        for condition in parsed['conditions']:
            cond_lower = condition.lower()
            if cond_lower in MEDICAL_ONTOLOGY:
                complications = MEDICAL_ONTOLOGY[cond_lower].get('complications', [])
                # Check if any complications are mentioned
                for comp in complications:
                    if comp in query_lower:
                        parsed['complications'].append(comp)
        
        return parsed
    
    def _is_complex_query(self, parsed: Dict[str, Any]) -> bool:
        """Determine if query requires multi-hop reasoning."""
        complexity_score = 0
        
        # Any condition mentioned
        if parsed['conditions']:
            complexity_score += 1
        
        # Age range specified
        if parsed['age_range']:
            complexity_score += 1
        
        # Gender specified
        if parsed['gender']:
            complexity_score += 1
        
        # Multiple conditions (bonus)
        if len(parsed['conditions']) >= 2:
            complexity_score += 1
        
        # Complications mentioned
        if parsed['complications']:
            complexity_score += 1
        
        # Severity specified
        if parsed['severity']:
            complexity_score += 1
        
        # Threshold lowered: any meaningful query triggers reasoning
        return complexity_score >= 1
    
    def _multi_hop_reasoning(
        self,
        query: str,
        parsed: Dict[str, Any]
    ) -> List[ReasoningStep]:
        """Perform multi-hop reasoning for complex queries."""
        steps = []
        step_num = 1
        
        # Step 1: Base query understanding (ALWAYS runs when multi-hop is enabled)
        if parsed['conditions']:
            conditions = parsed['conditions']
            if len(conditions) >= 2:
                step_query = f"What is the relationship between {' and '.join(conditions)} in patients?"
                description = f"Analyzing relationship between {' and '.join(conditions)}"
            else:
                condition = conditions[0]
                step_query = f"What are the key clinical characteristics and epidemiology of {condition}?"
                description = f"Understanding {condition} characteristics"
            
            result_summary = self._retrieve_and_summarize(step_query)
            
            steps.append(ReasoningStep(
                step_number=step_num,
                description=description,
                query=step_query,
                result_summary=result_summary['summary'],
                confidence=result_summary['confidence'],
                sources=result_summary['sources']
            ))
            step_num += 1
        
        # Step 2: Age-specific considerations
        if parsed['age_range']:
            age_desc = f"patients aged {parsed['age_range'][0]}-{parsed['age_range'][1]}"
            condition = parsed['conditions'][0] if parsed['conditions'] else 'general healthcare'
            
            step_query = f"What are the characteristics of {condition} in {age_desc}?"
            
            result_summary = self._retrieve_and_summarize(step_query)
            
            steps.append(ReasoningStep(
                step_number=step_num,
                description=f"Age-specific considerations for {age_desc}",
                query=step_query,
                result_summary=result_summary['summary'],
                confidence=result_summary['confidence'],
                sources=result_summary['sources']
            ))
            step_num += 1
        
        # Step 3: Gender-specific considerations
        if parsed['gender']:
            gender = parsed['gender']
            condition = parsed['conditions'][0] if parsed['conditions'] else 'healthcare outcomes'
            
            step_query = f"What are the gender-specific patterns of {condition} in {gender} patients?"
            
            result_summary = self._retrieve_and_summarize(step_query)
            
            steps.append(ReasoningStep(
                step_number=step_num,
                description=f"Gender-specific analysis for {gender} patients",
                query=step_query,
                result_summary=result_summary['summary'],
                confidence=result_summary['confidence'],
                sources=result_summary['sources']
            ))
            step_num += 1
        
        # Step 4: Complication patterns
        if parsed['complications']:
            comp = parsed['complications'][0]
            condition = parsed['conditions'][0] if parsed['conditions'] else 'general'
            
            step_query = f"What is the prevalence and pattern of {comp} in {condition} patients?"
            
            result_summary = self._retrieve_and_summarize(step_query)
            
            steps.append(ReasoningStep(
                step_number=step_num,
                description=f"Complication pattern: {comp}",
                query=step_query,
                result_summary=result_summary['summary'],
                confidence=result_summary['confidence'],
                sources=result_summary['sources']
            ))
            step_num += 1
        
        # Step 5: Severity considerations
        if parsed['severity']:
            severity = parsed['severity']
            condition = parsed['conditions'][0] if parsed['conditions'] else 'patient care'
            
            step_query = f"What defines {severity} severity in {condition} patients and outcomes?"
            
            result_summary = self._retrieve_and_summarize(step_query)
            
            steps.append(ReasoningStep(
                step_number=step_num,
                description=f"Severity analysis: {severity} cases",
                query=step_query,
                result_summary=result_summary['summary'],
                confidence=result_summary['confidence'],
                sources=result_summary['sources']
            ))
            step_num += 1
        
        # Final Step: Synthesize findings
        if steps:
            synthesis = self._synthesize_reasoning(steps)
            
            steps.append(ReasoningStep(
                step_number=step_num,
                description="Synthesis of findings for data generation",
                query="Synthesize findings",
                result_summary=synthesis,
                confidence=np.mean([s.confidence for s in steps]),
                sources=[]
            ))
        
        return steps
    
    def _retrieve_and_summarize(self, query: str) -> Dict[str, Any]:
        """Retrieve information and summarize."""
        result = {
            'summary': 'No information retrieved',
            'confidence': 0.5,
            'sources': []
        }
        
        if self.rag_system:
            try:
                rag_result = self.rag_system.extract_relevant_info(query)
                
                if rag_result.get('relevant_info'):
                    summaries = [info.get('content', '')[:200] for info in rag_result['relevant_info'][:3]]
                    result['summary'] = ' '.join(summaries) if summaries else rag_result.get('summary', '')
                    result['confidence'] = rag_result.get('confidence', 0.5)
                    
                    # Normalize sources to strings (RAG may return dicts or strings)
                    raw_sources = rag_result.get('sources', [])
                    result['sources'] = self._normalize_sources_to_strings(raw_sources)
                else:
                    result['summary'] = rag_result.get('summary', 'No specific information found')
                    result['confidence'] = rag_result.get('confidence', 0.3)
                    
            except Exception as e:
                logger.warning(f"RAG retrieval error: {e}")
                result['summary'] = f"Unable to retrieve: {str(e)}"
        
        return result
    
    def _normalize_sources_to_strings(self, sources: List[Any]) -> List[str]:
        """Convert sources (strings or dicts) to a list of strings."""
        normalized = []
        for src in sources:
            if isinstance(src, str):
                normalized.append(src)
            elif isinstance(src, dict):
                # Build a readable string from dict fields
                parts = []
                if src.get('source'):
                    parts.append(src['source'].upper())
                if src.get('title'):
                    parts.append(src['title'][:80])
                elif src.get('pmid'):
                    parts.append(f"PMID:{src['pmid']}")
                elif src.get('id'):
                    parts.append(f"ID:{src['id']}")
                if src.get('year'):
                    parts.append(f"({src['year']})")
                
                if parts:
                    normalized.append(' - '.join(parts))
                else:
                    # Fallback: stringify the dict
                    normalized.append(str(src))
            else:
                normalized.append(str(src))
        return normalized
    
    def _synthesize_reasoning(self, steps: List[ReasoningStep]) -> str:
        """Synthesize multi-hop reasoning steps into generation guidance."""
        synthesis_parts = []
        
        for step in steps:
            if step.result_summary and step.result_summary != 'No information retrieved':
                synthesis_parts.append(step.result_summary[:150])
        
        if synthesis_parts:
            return "Based on retrieved evidence: " + " | ".join(synthesis_parts)
        
        return "Generating based on domain knowledge and training data patterns"
    
    def _build_generation_context(
        self,
        query: str,
        parsed: Dict[str, Any],
        reasoning_chain: List[ReasoningStep]
    ) -> GenerationContext:
        """Build generation context from query and reasoning."""
        context = GenerationContext(
            conditions=parsed['conditions'],
            age_range=parsed['age_range'],
            gender=parsed['gender'],
            gender_ratio=parsed.get('gender_ratio', 1.0),
            severity=parsed['severity']
        )
        
        # Add constraints from reasoning
        for step in reasoning_chain:
            # Extract any numerical constraints from summaries
            constraints = self._extract_constraints_from_text(step.result_summary)
            context.constraints.extend(constraints)
            context.supporting_evidence.extend(step.sources)
        
        # Add ICU constraint if required
        if parsed['icu_required']:
            context.constraints.append(LiteratureConstraint(
                field='has_icu_stay',
                constraint_type='range',
                parameters={'min': 1, 'max': 1},
                source='Query requirement',
                confidence=1.0
            ))
        
        # Retrieve additional context using trainer
        if self.rag_system or self.enhanced_rag:
            trainer_context = self._trainer.retrieve_generation_context(
                query,
                parsed['conditions']
            )
            
            # Merge constraints
            for constraint in trainer_context.constraints:
                if constraint not in context.constraints:
                    context.constraints.append(constraint)
            
            context.retrieved_statistics.update(trainer_context.retrieved_statistics)
            context.supporting_evidence.extend(trainer_context.supporting_evidence)
        
        return context
    
    def _extract_constraints_from_text(self, text: str) -> List[LiteratureConstraint]:
        """Extract numerical constraints from retrieved text."""
        constraints = []
        
        if not text:
            return constraints
        
        # Pattern: X% prevalence of Y
        prevalence_pattern = r'(\d+(?:\.\d+)?)\s*%\s*(?:prevalence|rate)\s*(?:of\s+)?(\w+)'
        matches = re.findall(prevalence_pattern, text, re.IGNORECASE)
        
        for pct, condition in matches:
            try:
                constraints.append(LiteratureConstraint(
                    field=f'has_{condition.lower()}',
                    constraint_type='distribution',
                    parameters={'prevalence': float(pct) / 100},
                    source='Retrieved literature',
                    confidence=0.7
                ))
            except ValueError:
                pass
        
        # Pattern: mean age of X years
        age_pattern = r'mean\s+age\s+(?:of\s+)?(\d+(?:\.\d+)?)'
        age_matches = re.findall(age_pattern, text, re.IGNORECASE)
        
        for age in age_matches:
            try:
                constraints.append(LiteratureConstraint(
                    field='age',
                    constraint_type='distribution',
                    parameters={'mean': float(age), 'std': 10},
                    source='Retrieved literature',
                    confidence=0.7
                ))
            except ValueError:
                pass
        
        return constraints
    
    def _collect_citations(
        self,
        query: str,
        context: GenerationContext
    ) -> List[EvidenceCitation]:
        """Collect evidence citations from retrieval."""
        citations = []
        
        # From RAG system
        if self.rag_system:
            try:
                rag_result = self.rag_system.extract_relevant_info(query)
                
                for info in rag_result.get('relevant_info', [])[:5]:
                    citations.append(EvidenceCitation(
                        source=info.get('source', 'Unknown'),
                        source_type='literature',
                        title=info.get('title', info.get('source', '')),
                        relevance_score=info.get('relevance_score', 0.0),
                        excerpt=info.get('content', '')[:200] if info.get('content') else None
                    ))
                    
            except Exception as e:
                logger.warning(f"Error collecting citations from RAG: {e}")
        
        # From enhanced RAG
        if self.enhanced_rag:
            try:
                enhanced_result = self.enhanced_rag.retrieve_and_extract(
                    query,
                    max_results_per_source=5
                )
                
                for doc in enhanced_result.get('documents', [])[:5]:
                    source_type = doc.get('source', 'literature').lower()
                    if 'pubmed' in source_type:
                        source_type = 'pubmed'
                    elif 'clinical' in source_type or 'trial' in source_type:
                        source_type = 'clinical_trial'
                    elif 'who' in source_type:
                        source_type = 'who'
                    
                    citations.append(EvidenceCitation(
                        source=doc.get('source', 'Unknown'),
                        source_type=source_type,
                        title=doc.get('title', ''),
                        url=doc.get('url', ''),
                        relevance_score=doc.get('relevance_score', 0.0),
                        excerpt=doc.get('content', '')[:200] if doc.get('content') else None
                    ))
                    
            except Exception as e:
                logger.warning(f"Error collecting citations from enhanced RAG: {e}")
        
        # Deduplicate by source
        seen = set()
        unique_citations = []
        for c in citations:
            key = (c.source, c.title)
            if key not in seen:
                seen.add(key)
                unique_citations.append(c)
        
        return unique_citations[:10]
    
    def _fallback_generate(
        self,
        num_samples: int,
        context: GenerationContext
    ) -> pd.DataFrame:
        """Fallback generation when CTGAN is not available."""
        if self.original_dataset is None:
            raise ValueError("No CTGAN model or original dataset available for generation")
        
        # Filter original dataset based on context
        filtered = self.original_dataset.copy()
        
        # Apply condition filters
        for condition in context.conditions:
            condition_col = f'has_{condition.lower()}'
            if condition_col in filtered.columns:
                filtered = filtered[filtered[condition_col] == 1]
        
        # Apply age filter
        if context.age_range and 'age' in filtered.columns:
            min_age, max_age = context.age_range
            filtered = filtered[(filtered['age'] >= min_age) & (filtered['age'] <= max_age)]
        
        # Apply gender filter
        if context.gender and 'gender' in filtered.columns:
            filtered = filtered[filtered['gender'].str.lower() == context.gender.lower()]
        
        # Sample with replacement if needed
        if len(filtered) == 0:
            logger.warning("No matching records in original dataset, using random sample")
            return self.original_dataset.sample(n=num_samples, replace=True).reset_index(drop=True)
        
        return filtered.sample(n=min(num_samples, len(filtered) * 3), replace=True).head(num_samples).reset_index(drop=True)
    
    def _calculate_generation_confidence(
        self,
        data: pd.DataFrame,
        context: GenerationContext,
        reasoning_chain: List[ReasoningStep],
        citations: List[EvidenceCitation]
    ) -> float:
        """Calculate confidence score for generated data."""
        confidence = 0.5  # Base confidence
        
        # Boost for successful generation
        if len(data) > 0:
            confidence += 0.1
        
        # Boost for matching conditions
        conditions_matched = 0
        for condition in context.conditions:
            condition_col = f'has_{condition.lower()}'
            if condition_col in data.columns:
                if (data[condition_col] == 1).mean() > 0.8:
                    conditions_matched += 1
        
        if context.conditions and conditions_matched > 0:
            confidence += 0.1 * (conditions_matched / len(context.conditions))
        
        # Boost for literature support
        if citations:
            avg_relevance = np.mean([c.relevance_score for c in citations if c.relevance_score > 0])
            if avg_relevance > 0:
                confidence += min(avg_relevance * 0.2, 0.15)
        
        # Boost for reasoning chain
        if reasoning_chain:
            avg_reasoning_confidence = np.mean([s.confidence for s in reasoning_chain])
            confidence += min(avg_reasoning_confidence * 0.1, 0.1)
        
        # Penalize for small sample
        if len(data) < 10:
            confidence *= 0.8
        
        return min(max(confidence, 0.0), 1.0)
    
    def get_generation_history(self) -> List[Dict]:
        """Get history of generation requests."""
        return [r.to_dict() for r in self._generation_history]
