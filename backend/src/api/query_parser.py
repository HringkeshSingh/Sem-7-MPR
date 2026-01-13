"""
Enhanced Query Parser with Semantic Search.

Natural language query parser for healthcare data generation with:
- Semantic understanding using vector search
- Expanded medical terminology recognition
- Query expansion based on retrieved knowledge
- ICD-10 code mapping
- Medical ontology integration
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParsedEntity:
    """A parsed medical entity from the query."""
    entity_type: str  # 'condition', 'demographic', 'clinical', 'temporal'
    value: str
    normalized_value: str
    confidence: float
    source: str = 'keyword'  # 'keyword', 'semantic', 'ontology'
    icd10_codes: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence,
            "source": self.source,
            "icd10_codes": self.icd10_codes,
            "related_terms": self.related_terms
        }


@dataclass 
class ExpandedQuery:
    """Query with expansions and semantic understanding."""
    original_query: str
    normalized_query: str
    entities: List[ParsedEntity] = field(default_factory=list)
    expansions: List[str] = field(default_factory=list)
    semantic_concepts: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "entities": [e.to_dict() for e in self.entities],
            "expansions": self.expansions,
            "semantic_concepts": self.semantic_concepts,
            "filters": self.filters,
            "confidence": self.confidence
        }


# Comprehensive medical terminology with ICD-10 mappings
MEDICAL_TERMINOLOGY = {
    'DIABETES': {
        'keywords': [
            'diabetes', 'diabetic', 'dm', 'type 1 diabetes', 'type 2 diabetes',
            't1dm', 't2dm', 'glucose', 'insulin', 'hyperglycemia', 'hypoglycemia',
            'diabetes mellitus', 'blood sugar', 'glycemic', 'hba1c', 'a1c'
        ],
        'icd10': ['E10', 'E11', 'E13', 'E08', 'E09'],
        'related': ['diabetic nephropathy', 'diabetic retinopathy', 'diabetic neuropathy'],
        'complications': ['renal', 'cardiovascular', 'neurological', 'ophthalmological']
    },
    'CARDIOVASCULAR': {
        'keywords': [
            'cardiovascular', 'cardiac', 'heart', 'cvd', 'coronary', 'myocardial',
            'angina', 'heart attack', 'mi', 'myocardial infarction', 'cad',
            'coronary artery disease', 'heart failure', 'chf', 'cardiomyopathy',
            'arrhythmia', 'atrial fibrillation', 'afib', 'cardiac arrest'
        ],
        'icd10': ['I20', 'I21', 'I22', 'I25', 'I50', 'I48', 'I49'],
        'related': ['stent', 'bypass', 'cabg', 'pci', 'angioplasty'],
        'complications': ['heart_failure', 'arrhythmia', 'stroke']
    },
    'HYPERTENSION': {
        'keywords': [
            'hypertension', 'hypertensive', 'high blood pressure', 'htn', 'bp',
            'elevated blood pressure', 'essential hypertension', 'malignant hypertension',
            'resistant hypertension', 'secondary hypertension'
        ],
        'icd10': ['I10', 'I11', 'I12', 'I13', 'I15'],
        'related': ['blood pressure medication', 'antihypertensive'],
        'complications': ['stroke', 'cardiovascular', 'renal']
    },
    'RENAL': {
        'keywords': [
            'renal', 'kidney', 'nephritis', 'dialysis', 'ckd', 'chronic kidney',
            'acute kidney injury', 'aki', 'esrd', 'end stage renal', 'nephropathy',
            'glomerulonephritis', 'kidney failure', 'creatinine', 'gfr', 'bun',
            'hemodialysis', 'peritoneal dialysis'
        ],
        'icd10': ['N17', 'N18', 'N19', 'N00', 'N05'],
        'related': ['dialysis catheter', 'kidney transplant', 'fistula'],
        'complications': ['cardiovascular', 'anemia', 'bone_disease']
    },
    'RESPIRATORY': {
        'keywords': [
            'respiratory', 'copd', 'asthma', 'pneumonia', 'lung', 'bronchitis',
            'pulmonary', 'chronic obstructive', 'emphysema', 'ards', 'respiratory failure',
            'hypoxia', 'hypoxemia', 'oxygen', 'ventilator', 'intubation',
            'bronchiectasis', 'pulmonary fibrosis', 'pleural effusion'
        ],
        'icd10': ['J44', 'J45', 'J18', 'J96', 'J80'],
        'related': ['nebulizer', 'inhaler', 'oxygen therapy', 'cpap', 'bipap'],
        'complications': ['respiratory_failure', 'pulmonary_hypertension']
    },
    'SEPSIS': {
        'keywords': [
            'sepsis', 'septic', 'infection', 'bacteremia', 'septicemia',
            'septic shock', 'severe sepsis', 'urosepsis', 'systemic infection',
            'blood infection', 'sirs', 'qsofa'
        ],
        'icd10': ['A41', 'R65.2', 'A40'],
        'related': ['blood culture', 'antibiotics', 'source control'],
        'complications': ['organ_failure', 'dic', 'ards', 'aki']
    },
    'NEUROLOGICAL': {
        'keywords': [
            'neurological', 'stroke', 'seizure', 'brain', 'neuro', 'cerebral',
            'cva', 'cerebrovascular accident', 'tia', 'transient ischemic',
            'hemorrhagic stroke', 'ischemic stroke', 'epilepsy', 'encephalopathy',
            'meningitis', 'encephalitis', 'parkinson', 'alzheimer', 'dementia',
            'intracranial hemorrhage', 'ich', 'sah', 'subarachnoid'
        ],
        'icd10': ['I60', 'I61', 'I63', 'I64', 'G40', 'G20', 'G30'],
        'related': ['mri brain', 'ct head', 'lumbar puncture', 'eeg'],
        'complications': ['disability', 'cognitive_impairment']
    },
    'TRAUMA': {
        'keywords': [
            'trauma', 'fracture', 'injury', 'accident', 'wound', 'broken',
            'laceration', 'contusion', 'fall', 'mvc', 'motor vehicle',
            'head injury', 'tbi', 'traumatic brain injury', 'polytrauma',
            'blunt trauma', 'penetrating trauma', 'burns'
        ],
        'icd10': ['S00-S99', 'T07', 'T14'],
        'related': ['surgery', 'fixation', 'cast', 'splint'],
        'complications': ['infection', 'compartment_syndrome']
    },
    'CANCER': {
        'keywords': [
            'cancer', 'tumor', 'malignant', 'oncology', 'carcinoma', 'neoplasm',
            'metastatic', 'metastasis', 'chemotherapy', 'radiation', 'lymphoma',
            'leukemia', 'sarcoma', 'melanoma', 'adenocarcinoma', 'stage iv'
        ],
        'icd10': ['C00-C96', 'D00-D09'],
        'related': ['biopsy', 'oncologist', 'tumor marker'],
        'complications': ['metastasis', 'cachexia', 'immunosuppression']
    },
    'LIVER': {
        'keywords': [
            'liver', 'hepatic', 'cirrhosis', 'hepatitis', 'liver failure',
            'jaundice', 'ascites', 'hepatorenal', 'hepatocellular', 'hcc',
            'liver disease', 'fatty liver', 'nafld', 'nash', 'esophageal varices'
        ],
        'icd10': ['K70', 'K74', 'K72', 'B18'],
        'related': ['liver transplant', 'tips', 'paracentesis'],
        'complications': ['encephalopathy', 'variceal_bleeding', 'coagulopathy']
    },
    'PSYCHIATRIC': {
        'keywords': [
            'psychiatric', 'mental health', 'depression', 'anxiety', 'bipolar',
            'schizophrenia', 'psychosis', 'suicide', 'suicidal', 'overdose',
            'substance abuse', 'addiction', 'withdrawal', 'delirium'
        ],
        'icd10': ['F20', 'F30', 'F32', 'F41', 'F10-F19'],
        'related': ['psychiatrist', 'antidepressant', 'antipsychotic'],
        'complications': ['self_harm', 'medical_noncompliance']
    }
}

# Age-related terminology
AGE_TERMINOLOGY = {
    'elderly': {'range': (65, 100), 'synonyms': ['geriatric', 'senior', 'older adult', 'aged', 'old']},
    'very_elderly': {'range': (80, 100), 'synonyms': ['very old', 'oldest old', 'octogenarian', 'nonagenarian']},
    'middle_aged': {'range': (40, 65), 'synonyms': ['middle-aged', 'midlife']},
    'adult': {'range': (18, 65), 'synonyms': ['grown-up', 'mature']},
    'young_adult': {'range': (18, 40), 'synonyms': ['young', 'youth']},
    'adolescent': {'range': (12, 18), 'synonyms': ['teenager', 'teen', 'juvenile']},
    'pediatric': {'range': (0, 18), 'synonyms': ['child', 'children', 'kid', 'minor']},
    'infant': {'range': (0, 2), 'synonyms': ['baby', 'neonate', 'newborn']},
    'toddler': {'range': (1, 4), 'synonyms': ['preschool']}
}

# Clinical context terminology
CLINICAL_CONTEXT = {
    'icu': ['icu', 'intensive care', 'critical care', 'micu', 'sicu', 'ccu', 'nicu', 'picu'],
    'emergency': ['emergency', 'ed', 'er', 'urgent', 'acute', 'emergent', 'trauma center'],
    'inpatient': ['inpatient', 'admitted', 'hospitalized', 'hospital stay', 'ward'],
    'outpatient': ['outpatient', 'clinic', 'ambulatory', 'office visit'],
    'surgical': ['surgical', 'surgery', 'operative', 'post-op', 'pre-op', 'or'],
    'palliative': ['palliative', 'hospice', 'end of life', 'comfort care', 'dnr']
}


class EnhancedQueryParser:
    """
    Enhanced query parser with semantic understanding and medical terminology expansion.
    
    Usage:
        parser = EnhancedQueryParser()
        
        # Basic parsing
        filters = parser.parse_query("Generate 100 elderly diabetic patients")
        
        # With semantic expansion
        parser.set_vector_search(vector_search_client)
        parser.set_rag_system(rag_system)
        
        expanded = parser.expand_query("elderly patients with heart problems")
        print(expanded.expansions)  # ['cardiovascular disease', 'heart failure', ...]
    """
    
    def __init__(
        self,
        vector_search: Optional[Any] = None,
        rag_system: Optional[Any] = None
    ):
        self.terminology = MEDICAL_TERMINOLOGY
        self.age_terms = AGE_TERMINOLOGY
        self.clinical_context = CLINICAL_CONTEXT
        
        self._vector_search = vector_search
        self._rag_system = rag_system
        
        # Build reverse lookup for faster matching
        self._keyword_to_condition = self._build_keyword_lookup()
    
    def set_vector_search(self, vector_search: Any):
        """Set vector search client for semantic queries."""
        self._vector_search = vector_search
    
    def set_rag_system(self, rag_system: Any):
        """Set RAG system for knowledge-based expansion."""
        self._rag_system = rag_system
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query into structured filters.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary of parsed filters
        """
        query_lower = query.lower().strip()
        
        filters = {
            'diagnoses': [],
            'diagnosis_logic': 'OR',
            'age_range': None,
            'gender': None,
            'icu_required': None,
            'mortality': None,
            'risk_level': None,
            'complexity': None,
            'emergency': None,
            'clinical_context': None,
            'length_of_stay': None,
            'sample_size': None,
            'entities': []
        }
        
        # Extract diagnoses with expanded matching
        diagnoses, entities = self._extract_diagnoses_enhanced(query_lower)
        filters['diagnoses'] = diagnoses
        filters['entities'] = entities
        
        # Detect AND vs OR logic
        if len(filters['diagnoses']) > 1:
            if ' and ' in query_lower or ' with ' in query_lower:
                filters['diagnosis_logic'] = 'AND'
            elif ' or ' in query_lower:
                filters['diagnosis_logic'] = 'OR'
        
        # Extract age information with enhanced patterns
        filters['age_range'] = self._extract_age_range_enhanced(query_lower)
        
        # Extract gender
        filters['gender'] = self._extract_gender(query_lower)
        
        # Extract clinical context
        filters['clinical_context'] = self._extract_clinical_context(query_lower)
        
        # Extract ICU requirement
        filters['icu_required'] = self._extract_icu_requirement(query_lower)
        
        # Extract mortality information
        filters['mortality'] = self._extract_mortality(query_lower)
        
        # Extract risk level
        filters['risk_level'] = self._extract_risk_level(query_lower)
        
        # Extract complexity
        filters['complexity'] = self._extract_complexity(query_lower)
        
        # Extract emergency status
        filters['emergency'] = self._extract_emergency_status(query_lower)
        
        # Extract length of stay preferences
        filters['length_of_stay'] = self._extract_length_of_stay(query_lower)
        
        # Extract sample size
        filters['sample_size'] = self._extract_count(query_lower)
        
        logger.debug(f"Parsed query '{query}' into filters: {filters}")
        return filters
    
    def expand_query(self, query: str) -> ExpandedQuery:
        """
        Expand query with semantic understanding and related terms.
        
        Args:
            query: Natural language query string
            
        Returns:
            ExpandedQuery with expansions and semantic concepts
        """
        query_lower = query.lower().strip()
        
        # Start with basic parsing
        filters = self.parse_query(query)
        
        entities = []
        expansions = []
        semantic_concepts = []
        
        # Extract and expand medical entities
        for diagnosis in filters['diagnoses']:
            if diagnosis in self.terminology:
                term_info = self.terminology[diagnosis]
                
                entity = ParsedEntity(
                    entity_type='condition',
                    value=diagnosis,
                    normalized_value=diagnosis,
                    confidence=0.9,
                    source='keyword',
                    icd10_codes=term_info.get('icd10', []),
                    related_terms=term_info.get('related', [])
                )
                entities.append(entity)
                
                # Add expansions
                expansions.extend(term_info.get('related', [])[:3])
                
                # Add complications as concepts
                semantic_concepts.extend(term_info.get('complications', []))
        
        # Semantic expansion using vector search
        if self._vector_search:
            semantic_results = self._semantic_expand(query)
            for concept in semantic_results:
                if concept not in semantic_concepts:
                    semantic_concepts.append(concept)
        
        # Knowledge-based expansion using RAG
        if self._rag_system:
            rag_expansions = self._rag_expand(query)
            for exp in rag_expansions:
                if exp not in expansions:
                    expansions.append(exp)
        
        # Normalize the query
        normalized = self._normalize_query(query_lower, entities)
        
        # Calculate confidence
        confidence = self._calculate_parse_confidence(filters, entities)
        
        return ExpandedQuery(
            original_query=query,
            normalized_query=normalized,
            entities=entities,
            expansions=expansions[:10],  # Limit expansions
            semantic_concepts=semantic_concepts[:10],
            filters=filters,
            confidence=confidence
        )
    
    def get_icd10_codes(self, query: str) -> List[str]:
        """Extract ICD-10 codes for conditions in query."""
        filters = self.parse_query(query)
        codes = []
        
        for diagnosis in filters['diagnoses']:
            if diagnosis in self.terminology:
                codes.extend(self.terminology[diagnosis].get('icd10', []))
        
        return list(set(codes))
    
    def suggest_related_queries(self, query: str, max_suggestions: int = 5) -> List[str]:
        """Suggest related queries based on parsed entities."""
        expanded = self.expand_query(query)
        suggestions = []
        
        for diagnosis in expanded.filters.get('diagnoses', []):
            if diagnosis in self.terminology:
                term_info = self.terminology[diagnosis]
                
                # Suggest queries with complications
                for comp in term_info.get('complications', [])[:2]:
                    comp_name = comp.replace('_', ' ')
                    suggestions.append(
                        f"Generate patients with {diagnosis.lower()} and {comp_name}"
                    )
                
                # Suggest with age group
                suggestions.append(
                    f"Generate elderly patients with {diagnosis.lower()}"
                )
        
        return suggestions[:max_suggestions]
    
    def _build_keyword_lookup(self) -> Dict[str, str]:
        """Build reverse lookup from keywords to conditions."""
        lookup = {}
        
        for condition, info in self.terminology.items():
            for keyword in info['keywords']:
                lookup[keyword.lower()] = condition
        
        return lookup
    
    def _extract_diagnoses_enhanced(self, query: str) -> Tuple[List[str], List[ParsedEntity]]:
        """Extract diagnoses with enhanced matching and entity creation."""
        found_diagnoses = []
        entities = []
        
        # Check each keyword
        for keyword, condition in self._keyword_to_condition.items():
            # Use word boundary matching for better accuracy
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, query):
                if condition not in found_diagnoses:
                    found_diagnoses.append(condition)
                    
                    entities.append(ParsedEntity(
                        entity_type='condition',
                        value=keyword,
                        normalized_value=condition,
                        confidence=0.9 if len(keyword) > 3 else 0.7,
                        source='keyword',
                        icd10_codes=self.terminology[condition].get('icd10', [])
                    ))
        
        return found_diagnoses, entities
    
    def _extract_age_range_enhanced(self, query: str) -> Optional[Tuple[int, int]]:
        """Extract age range with enhanced pattern matching."""
        # Check named age groups first
        for age_group, info in self.age_terms.items():
            all_terms = [age_group.replace('_', ' ')] + info['synonyms']
            for term in all_terms:
                if term in query:
                    return info['range']
        
        # Check for explicit age ranges
        patterns = [
            (r'(\d+)\s*[-–to]+\s*(\d+)\s*(?:years?|y/?o)?', lambda m: (int(m.group(1)), int(m.group(2)))),
            (r'(?:over|above|older than|>\s*)\s*(\d+)', lambda m: (int(m.group(1)), 100)),
            (r'(?:under|below|younger than|<\s*)\s*(\d+)', lambda m: (0, int(m.group(1)))),
            (r'(?:age[sd]?|aged)\s*(\d+)', lambda m: (int(m.group(1)) - 5, int(m.group(1)) + 5)),
            (r'(\d+)\s*(?:year|yr)s?\s*old', lambda m: (int(m.group(1)) - 2, int(m.group(1)) + 2)),
        ]
        
        for pattern, extractor in patterns:
            match = re.search(pattern, query)
            if match:
                try:
                    result = extractor(match)
                    # Validate range
                    min_age = max(0, min(result[0], 120))
                    max_age = max(min_age, min(result[1], 120))
                    return (min_age, max_age)
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_gender(self, query: str) -> Optional[str]:
        """Extract gender preference from query."""
        male_terms = ['male', 'men', 'man', 'boy', 'gentleman']
        female_terms = ['female', 'women', 'woman', 'girl', 'lady']
        
        for term in male_terms:
            if re.search(r'\b' + term + r'\b', query):
                return 'male'
        
        for term in female_terms:
            if re.search(r'\b' + term + r'\b', query):
                return 'female'
        
        return None
    
    def _extract_clinical_context(self, query: str) -> Optional[str]:
        """Extract clinical context from query."""
        for context, keywords in self.clinical_context.items():
            if any(kw in query for kw in keywords):
                return context
        return None
    
    def _extract_icu_requirement(self, query: str) -> Optional[bool]:
        """Extract ICU requirement from query."""
        if any(term in query for term in self.clinical_context['icu']):
            return True
        
        non_icu_terms = ['non-icu', 'non icu', 'general ward', 'regular ward', 'floor patient']
        if any(term in query for term in non_icu_terms):
            return False
        
        return None
    
    def _extract_mortality(self, query: str) -> Optional[bool]:
        """Extract mortality information from query."""
        death_terms = ['died', 'death', 'mortality', 'fatal', 'deceased', 'expired', 'non-survivor']
        survival_terms = ['survived', 'survivor', 'discharged', 'recovered', 'alive']
        
        if any(term in query for term in death_terms):
            return True
        elif any(term in query for term in survival_terms):
            return False
        
        return None
    
    def _extract_risk_level(self, query: str) -> Optional[str]:
        """Extract risk level from query."""
        risk_mapping = {
            'critical': ['critical', 'severe', 'life-threatening', 'unstable'],
            'high': ['high risk', 'high-risk', 'serious', 'major', 'significant'],
            'medium': ['moderate', 'medium', 'intermediate', 'average'],
            'low': ['mild', 'minor', 'low risk', 'low-risk', 'stable']
        }
        
        for level, terms in risk_mapping.items():
            if any(term in query for term in terms):
                return level
        
        return None
    
    def _extract_complexity(self, query: str) -> Optional[str]:
        """Extract complexity level from query."""
        high_terms = ['complex', 'complicated', 'multiple', 'comorbid', 'multi-diagnosis']
        low_terms = ['simple', 'single', 'straightforward', 'uncomplicated']
        
        if any(term in query for term in high_terms):
            return 'high'
        elif any(term in query for term in low_terms):
            return 'low'
        
        return None
    
    def _extract_emergency_status(self, query: str) -> Optional[bool]:
        """Extract emergency admission status from query."""
        if any(term in query for term in self.clinical_context['emergency']):
            return True
        
        elective_terms = ['elective', 'planned', 'scheduled', 'routine']
        if any(term in query for term in elective_terms):
            return False
        
        return None
    
    def _extract_length_of_stay(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract length of stay preferences from query."""
        los_info = {}
        
        # Look for specific day mentions
        days_match = re.search(r'(\d+)\s*days?', query)
        if days_match:
            los_info['target_days'] = int(days_match.group(1))
        
        # Duration indicators
        if any(term in query for term in ['short stay', 'brief', 'quick']):
            los_info['duration'] = 'short'
        elif any(term in query for term in ['long stay', 'extended', 'prolonged']):
            los_info['duration'] = 'long'
        
        return los_info if los_info else None
    
    def _extract_count(self, query: str) -> Optional[int]:
        """Extract the number of records requested from query."""
        patterns = [
            r'(\d+)\s*patients?',
            r'(\d+)\s*records?',
            r'(\d+)\s*cases?',
            r'generate\s*(\d+)',
            r'create\s*(\d+)',
            r'(\d+)\s*samples?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                count = int(match.group(1))
                return min(max(count, 1), 10000)
        
        return None
    
    def _semantic_expand(self, query: str) -> List[str]:
        """Use vector search for semantic query expansion."""
        if not self._vector_search:
            return []
        
        try:
            # Search for similar medical concepts
            results = self._vector_search.search(
                query=query,
                k=5,
                search_type='semantic'
            )
            
            concepts = []
            for result in results.get('results', []):
                content = result.get('content', '')
                # Extract key medical terms from content
                for condition in self.terminology:
                    if condition.lower() in content.lower():
                        concepts.append(condition)
            
            return list(set(concepts))
            
        except Exception as e:
            logger.warning(f"Semantic expansion error: {e}")
            return []
    
    def _rag_expand(self, query: str) -> List[str]:
        """Use RAG system for knowledge-based query expansion."""
        if not self._rag_system:
            return []
        
        try:
            result = self._rag_system.extract_relevant_info(query)
            
            expansions = []
            if result.get('relevant_info'):
                for info in result['relevant_info'][:3]:
                    content = info.get('content', '')
                    
                    # Extract related medical terms
                    for condition, info_dict in self.terminology.items():
                        for keyword in info_dict['keywords'][:3]:
                            if keyword in content.lower():
                                if condition not in expansions:
                                    expansions.append(condition)
            
            return expansions
            
        except Exception as e:
            logger.warning(f"RAG expansion error: {e}")
            return []
    
    def _normalize_query(self, query: str, entities: List[ParsedEntity]) -> str:
        """Normalize query by replacing variations with standard terms."""
        normalized = query
        
        for entity in entities:
            if entity.value != entity.normalized_value:
                # Replace the original term with normalized version
                normalized = re.sub(
                    r'\b' + re.escape(entity.value) + r'\b',
                    entity.normalized_value.lower(),
                    normalized
                )
        
        return normalized
    
    def _calculate_parse_confidence(
        self,
        filters: Dict[str, Any],
        entities: List[ParsedEntity]
    ) -> float:
        """Calculate confidence in parsing results."""
        confidence = 0.5  # Base confidence
        
        # Boost for each recognized entity
        if filters['diagnoses']:
            confidence += min(len(filters['diagnoses']) * 0.1, 0.2)
        
        if filters['age_range']:
            confidence += 0.1
        
        if filters['gender']:
            confidence += 0.05
        
        if filters['clinical_context']:
            confidence += 0.1
        
        # Boost for high-confidence entities
        if entities:
            avg_entity_conf = sum(e.confidence for e in entities) / len(entities)
            confidence += avg_entity_conf * 0.1
        
        return min(confidence, 1.0)
    
    def validate_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean the parsed filters."""
        validated = {}
        
        # Validate diagnoses
        if filters.get('diagnoses'):
            validated['diagnoses'] = [d for d in filters['diagnoses'] if d in self.terminology]
            validated['diagnosis_logic'] = filters.get('diagnosis_logic', 'OR')
        
        # Validate age range
        if filters.get('age_range'):
            min_age, max_age = filters['age_range']
            min_age = max(0, min_age)
            max_age = min(120, max_age)
            if min_age < max_age:
                validated['age_range'] = (min_age, max_age)
        
        # Validate gender
        if filters.get('gender') in ['male', 'female']:
            validated['gender'] = filters['gender']
        
        # Validate boolean fields
        for field in ['icu_required', 'mortality', 'emergency']:
            if filters.get(field) is not None:
                validated[field] = bool(filters[field])
        
        # Validate categorical fields
        if filters.get('risk_level') in ['low', 'medium', 'high', 'critical']:
            validated['risk_level'] = filters['risk_level']
        
        if filters.get('complexity') in ['low', 'medium', 'high']:
            validated['complexity'] = filters['complexity']
        
        if filters.get('clinical_context'):
            validated['clinical_context'] = filters['clinical_context']
        
        if filters.get('sample_size'):
            validated['sample_size'] = min(max(1, filters['sample_size']), 10000)
        
        return validated
    
    def generate_example_queries(self) -> List[str]:
        """Generate example queries for API documentation."""
        return [
            "Generate 100 elderly patients with diabetes and hypertension",
            "Create 50 ICU patients with cardiovascular disease",
            "Generate young adults with respiratory conditions who survived",
            "Create critical care patients with multiple comorbidities",
            "Generate 200 emergency admission patients over 70",
            "Create elderly female patients with renal disease requiring dialysis",
            "Generate sepsis patients with organ failure",
            "Create complex cases with diabetes and cardiovascular complications",
            "Generate trauma patients from motor vehicle accidents",
            "Create high-risk cardiac patients with heart failure"
        ]


# Backward compatibility alias
HealthcareQueryParser = EnhancedQueryParser
