"""
Content Preprocessor.

Cleans, normalizes, and enriches retrieved content:
- Text cleaning and normalization
- Medical entity extraction (diseases, drugs, procedures)
- Duplicate detection and removal
- Quality scoring
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# Medical entity patterns
DRUG_PATTERNS = [
    r'\b[A-Z][a-z]+(?:mab|nib|vir|pril|sartan|statin|olol|dipine|zole|mycin|cillin|floxacin)\b',
    r'\b(?:aspirin|ibuprofen|metformin|insulin|warfarin|lisinopril|atorvastatin|omeprazole|losartan|amlodipine)\b'
]

DISEASE_PATTERNS = {
    "diabetes": r'\b(?:diabetes|diabetic|t1dm|t2dm|hyperglycemia|hypoglycemia|insulin.?resistance)\b',
    "cardiovascular": r'\b(?:cardiac|heart|coronary|cardiovascular|myocardial|angina|arrhythmia|atrial.?fibrillation)\b',
    "respiratory": r'\b(?:respiratory|pulmonary|lung|copd|asthma|pneumonia|bronchitis|emphysema)\b',
    "neurological": r'\b(?:neurological|brain|stroke|alzheimer|parkinson|epilepsy|seizure|dementia)\b',
    "oncology": r'\b(?:cancer|tumor|oncology|carcinoma|malignant|metastasis|chemotherapy|radiation)\b',
    "infectious": r'\b(?:infection|sepsis|bacterial|viral|antibiotic|pathogen|fever)\b',
    "renal": r'\b(?:kidney|renal|nephritis|dialysis|creatinine|proteinuria)\b',
    "hypertension": r'\b(?:hypertension|high.?blood.?pressure|hypotension)\b'
}

PROCEDURE_PATTERNS = [
    r'\b(?:surgery|surgical|transplant|biopsy|endoscopy|catheterization|angioplasty|bypass)\b',
    r'\b(?:MRI|CT.?scan|X.?ray|ultrasound|echocardiogram|EKG|ECG)\b',
    r'\b(?:dialysis|chemotherapy|radiation.?therapy|immunotherapy|ventilation)\b'
]

MEASUREMENT_PATTERNS = [
    r'\b(\d+(?:\.\d+)?)\s*(?:mg|kg|ml|mmHg|mmol|%|bpm|mm)\b',
    r'\b(?:HbA1c|BMI|GFR|eGFR|LDL|HDL|blood.?pressure)\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\b'
]


@dataclass
class ExtractedEntity:
    """Extracted medical entity."""
    text: str
    entity_type: str  # drug, disease, procedure, measurement
    category: Optional[str] = None
    confidence: float = 1.0
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class ProcessedContent:
    """Processed and enriched content."""
    original_text: str
    cleaned_text: str
    entities: List[ExtractedEntity] = field(default_factory=list)
    diseases: List[str] = field(default_factory=list)
    drugs: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0
    word_count: int = 0
    hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cleaned_text": self.cleaned_text,
            "entities": [{"text": e.text, "type": e.entity_type, "category": e.category} 
                        for e in self.entities],
            "diseases": self.diseases,
            "drugs": self.drugs,
            "procedures": self.procedures,
            "measurements": self.measurements,
            "quality_score": self.quality_score,
            "word_count": self.word_count
        }


class ContentProcessor:
    """
    Content preprocessor for medical text.
    
    Usage:
        processor = ContentProcessor()
        
        # Process single text
        result = processor.process("Patient with diabetes treated with metformin...")
        
        # Process and deduplicate batch
        unique_docs = processor.process_batch(documents, deduplicate=True)
    """
    
    def __init__(
        self,
        min_quality_score: float = 0.3,
        min_word_count: int = 20,
        extract_entities: bool = True
    ):
        self.min_quality_score = min_quality_score
        self.min_word_count = min_word_count
        self.extract_entities = extract_entities
        self._seen_hashes: Set[str] = set()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove HTML tags (only if text contains HTML)
        if '<' in text and '>' in text:
            text = BeautifulSoup(text, "html.parser").get_text()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical punctuation
        text = re.sub(r'[^\w\s\-.,;:()/%]', '', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:])', r'\1', text)
        text = re.sub(r'([.,;:])\s*', r'\1 ', text)
        
        return text.strip()
    
    def extract_drugs(self, text: str) -> List[str]:
        """Extract drug names from text."""
        drugs = set()
        
        for pattern in DRUG_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            drugs.update(m.lower() for m in matches)
        
        return sorted(drugs)
    
    def extract_diseases(self, text: str) -> Tuple[List[str], List[ExtractedEntity]]:
        """Extract disease mentions from text."""
        diseases = []
        entities = []
        text_lower = text.lower()
        
        for category, pattern in DISEASE_PATTERNS.items():
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if category not in diseases:
                    diseases.append(category)
                
                entities.append(ExtractedEntity(
                    text=match.group(),
                    entity_type="disease",
                    category=category,
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        return diseases, entities
    
    def extract_procedures(self, text: str) -> List[str]:
        """Extract medical procedures from text."""
        procedures = set()
        
        for pattern in PROCEDURE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            procedures.update(m.lower() for m in matches)
        
        return sorted(procedures)
    
    def extract_measurements(self, text: str) -> List[Dict[str, Any]]:
        """Extract clinical measurements from text."""
        measurements = []
        
        for pattern in MEASUREMENT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value = match.group(1) if match.lastindex else match.group()
                    measurements.append({
                        "text": match.group(),
                        "value": float(value) if value else None
                    })
                except (ValueError, IndexError):
                    continue
        
        return measurements
    
    def calculate_quality_score(self, text: str, entities: List[ExtractedEntity]) -> float:
        """Calculate content quality score."""
        score = 0.0
        
        # Word count factor
        words = text.split()
        word_count = len(words)
        
        if word_count < 10:
            score += 0.0
        elif word_count < 50:
            score += 0.2
        elif word_count < 200:
            score += 0.3
        else:
            score += 0.35
        
        # Entity density
        if entities:
            entity_density = len(entities) / max(word_count, 1)
            score += min(entity_density * 10, 0.25)
        
        # Sentence structure
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) >= 3:
            score += 0.1
        
        # Medical terminology presence
        medical_terms = sum(1 for pattern in DISEASE_PATTERNS.values() 
                          if re.search(pattern, text.lower()))
        score += min(medical_terms * 0.05, 0.2)
        
        # Readability (average word length as proxy)
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        if 4 <= avg_word_len <= 8:
            score += 0.1
        
        return min(score, 1.0)
    
    def compute_hash(self, text: str) -> str:
        """Compute content hash for deduplication."""
        # Normalize text for hashing
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def process(self, text: str) -> ProcessedContent:
        """Process a single text document."""
        # Clean text
        cleaned = self.clean_text(text)
        
        # Extract entities
        entities = []
        diseases = []
        drugs = []
        procedures = []
        measurements = []
        
        if self.extract_entities and cleaned:
            diseases, disease_entities = self.extract_diseases(cleaned)
            entities.extend(disease_entities)
            
            drugs = self.extract_drugs(cleaned)
            for drug in drugs:
                entities.append(ExtractedEntity(
                    text=drug,
                    entity_type="drug",
                    confidence=0.8
                ))
            
            procedures = self.extract_procedures(cleaned)
            for proc in procedures:
                entities.append(ExtractedEntity(
                    text=proc,
                    entity_type="procedure",
                    confidence=0.85
                ))
            
            measurements = self.extract_measurements(cleaned)
        
        # Calculate quality
        quality_score = self.calculate_quality_score(cleaned, entities)
        word_count = len(cleaned.split())
        content_hash = self.compute_hash(cleaned)
        
        return ProcessedContent(
            original_text=text,
            cleaned_text=cleaned,
            entities=entities,
            diseases=diseases,
            drugs=drugs,
            procedures=procedures,
            measurements=measurements,
            quality_score=quality_score,
            word_count=word_count,
            hash=content_hash
        )
    
    def process_batch(
        self,
        texts: List[str],
        deduplicate: bool = True,
        filter_low_quality: bool = True
    ) -> List[ProcessedContent]:
        """
        Process a batch of texts with optional deduplication.
        
        Args:
            texts: List of texts to process
            deduplicate: Remove duplicate content
            filter_low_quality: Remove low-quality content
        """
        results = []
        seen_hashes = set()
        
        for text in texts:
            processed = self.process(text)
            
            # Skip duplicates
            if deduplicate and processed.hash in seen_hashes:
                continue
            
            # Skip low quality
            if filter_low_quality:
                if processed.quality_score < self.min_quality_score:
                    continue
                if processed.word_count < self.min_word_count:
                    continue
            
            seen_hashes.add(processed.hash)
            results.append(processed)
        
        logger.info(f"Processed {len(texts)} texts -> {len(results)} unique, quality documents")
        return results
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate of previously seen content."""
        content_hash = self.compute_hash(text)
        
        if content_hash in self._seen_hashes:
            return True
        
        self._seen_hashes.add(content_hash)
        return False
    
    def clear_seen(self):
        """Clear seen content hashes."""
        self._seen_hashes.clear()
    
    def extract_all_entities(self, texts: List[str]) -> Dict[str, List[str]]:
        """Extract all unique entities from a list of texts."""
        all_diseases = set()
        all_drugs = set()
        all_procedures = set()
        
        for text in texts:
            cleaned = self.clean_text(text)
            diseases, _ = self.extract_diseases(cleaned)
            all_diseases.update(diseases)
            all_drugs.update(self.extract_drugs(cleaned))
            all_procedures.update(self.extract_procedures(cleaned))
        
        return {
            "diseases": sorted(all_diseases),
            "drugs": sorted(all_drugs),
            "procedures": sorted(all_procedures)
        }
    
    def summarize_entities(self, processed_docs: List[ProcessedContent]) -> Dict[str, Any]:
        """Summarize entities across processed documents."""
        disease_counts = defaultdict(int)
        drug_counts = defaultdict(int)
        procedure_counts = defaultdict(int)
        
        for doc in processed_docs:
            for disease in doc.diseases:
                disease_counts[disease] += 1
            for drug in doc.drugs:
                drug_counts[drug] += 1
            for proc in doc.procedures:
                procedure_counts[proc] += 1
        
        return {
            "total_documents": len(processed_docs),
            "avg_quality_score": sum(d.quality_score for d in processed_docs) / max(len(processed_docs), 1),
            "disease_distribution": dict(sorted(disease_counts.items(), key=lambda x: -x[1])),
            "top_drugs": dict(sorted(drug_counts.items(), key=lambda x: -x[1])[:20]),
            "top_procedures": dict(sorted(procedure_counts.items(), key=lambda x: -x[1])[:10])
        }
