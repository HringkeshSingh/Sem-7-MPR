"""
rag_utils.py

Utility functions for the RAG system.
Helper functions for summary generation, confidence calculation, and parameter filtering.
"""

import re
from typing import List, Dict, Any

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# Healthcare condition keywords mapping
CONDITION_KEYWORDS = {
    'diabetes': 'DIABETES',
    'cardiovascular': 'CARDIOVASCULAR',
    'hypertension': 'HYPERTENSION',
    'renal': 'RENAL',
    'respiratory': 'RESPIRATORY',
    'sepsis': 'SEPSIS',
    'neurological': 'NEUROLOGICAL',
    'trauma': 'TRAUMA',
    'cancer': 'CANCER'
}

# Clinical factors to extract
CLINICAL_FACTORS = ['icu', 'mortality', 'severity', 'risk', 'comorbidity']


def generate_summary(query: str, relevant_info: List[Dict[str, Any]]) -> str:
    """
    Generate a human-readable summary of relevant information.
    
    Extracts key points from the top relevant documents and formats them
    into a readable summary.
    
    Args:
        query: Original user query
        relevant_info: List of relevant information dictionaries
        
    Returns:
        Formatted summary string
    """
    if not relevant_info:
        return "No relevant information found."
    
    # Extract key points from top 3 most relevant documents
    key_points = []
    for info in relevant_info[:3]:
        content = info.get('content', '')
        # Extract first 2 sentences for brevity
        sentences = content.split('. ')[:2]
        key_point = '. '.join(sentences)
        if key_point.strip():
            key_points.append(key_point)
    
    if not key_points:
        return f"No extractable information found for query: '{query}'"
    
    # Format summary
    summary = f"Based on the query '{query}', the following relevant information was found:\n\n"
    summary += "\n\n".join([f"- {point}" for point in key_points])
    
    return summary


def calculate_confidence(relevant_info: List[Dict[str, Any]]) -> float:
    """
    Calculate confidence score based on relevance scores and document count.
    
    Confidence is calculated as a combination of:
    1. Average relevance scores (if available)
    2. Number of documents found (more = higher confidence)
    3. Content quality (non-empty content)
    
    Args:
        relevant_info: List of relevant information dictionaries
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not relevant_info:
        return 0.0
    
    # Extract relevance scores
    scores = []
    content_quality = 0
    
    for info in relevant_info:
        score = info.get('relevance_score', 0.0)
        scores.append(score)
        
        # Check content quality (non-empty, meaningful length)
        content = info.get('content', '')
        if content and len(content.strip()) > 50:
            content_quality += 1
    
    # Calculate base confidence from scores
    if scores and not all(s == 0.0 for s in scores):
        avg_score = sum(scores) / len(scores)
    else:
        # If no meaningful scores, use position-based default
        # First document gets higher weight
        avg_score = 0.7 if len(relevant_info) > 0 else 0.0
    
    # Factor in document count (more documents = higher confidence, up to a point)
    count_factor = min(len(relevant_info) / 5.0, 1.0) * 0.2
    
    # Factor in content quality
    quality_factor = (content_quality / len(relevant_info)) * 0.1 if relevant_info else 0.0
    
    # Combine factors
    confidence = min(max(avg_score + count_factor + quality_factor, 0.0), 1.0)
    
    # Ensure minimum confidence if we have documents
    if relevant_info and confidence < 0.3:
        confidence = 0.3 + (len(relevant_info) * 0.1)
        confidence = min(confidence, 1.0)
    
    return confidence


def filter_query_parameters(
    query: str,
    extracted_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Filter and extract relevant parameters from query based on retrieved information.
    
    Analyzes retrieved documents to extract:
    - Medical conditions mentioned
    - Demographic information (age, gender)
    - Clinical factors (ICU, mortality, etc.)
    
    Args:
        query: Original user query
        extracted_info: Information extracted by RAG system
        
    Returns:
        Dictionary containing filtered parameters:
        - original_query: Original query string
        - relevant_conditions: List of medical conditions found
        - relevant_demographics: Dictionary with age/gender info
        - relevant_clinical_factors: List of clinical factors
        - confidence: Confidence score from extraction
    """
    try:
        filtered_params = {
            'original_query': query,
            'relevant_conditions': [],
            'relevant_demographics': {},
            'relevant_clinical_factors': [],
            'confidence': extracted_info.get('confidence', 0.0)
        }
        
        # Get relevant documents
        relevant_docs = extracted_info.get('relevant_info', [])
        
        if not relevant_docs:
            logger.debug("No relevant documents to extract parameters from")
            return filtered_params
        
        # Process each document
        for doc in relevant_docs:
            content = doc.get('content', '').lower()
            metadata = doc.get('metadata', {})
            
            # Extract medical conditions
            _extract_conditions(content, filtered_params)
            
            # Extract demographic information
            _extract_demographics(content, filtered_params)
            
            # Extract clinical factors
            _extract_clinical_factors(content, filtered_params)
        
        logger.debug(f"Filtered parameters: {filtered_params}")
        return filtered_params
        
    except Exception as e:
        logger.error(f"Error filtering query parameters: {e}")
        return {
            'original_query': query,
            'relevant_conditions': [],
            'relevant_demographics': {},
            'relevant_clinical_factors': [],
            'confidence': 0.0,
            'error': str(e)
        }


def _extract_conditions(content: str, filtered_params: Dict[str, Any]):
    """Extract medical conditions from document content."""
    for keyword, condition in CONDITION_KEYWORDS.items():
        if keyword in content and condition not in filtered_params['relevant_conditions']:
            filtered_params['relevant_conditions'].append(condition)


def _extract_demographics(content: str, filtered_params: Dict[str, Any]):
    """Extract demographic information (age, gender) from document content."""
    # Check for age-related keywords
    age_keywords = ['age', 'elderly', 'young', 'adult', 'pediatric', 'geriatric']
    has_age_info = any(keyword in content for keyword in age_keywords)
    
    if has_age_info and 'age_range' not in filtered_params['relevant_demographics']:
        # Try to extract specific age range
        age_match = re.search(r'age[ds]?\s*(\d+)\s*[-to]?\s*(\d+)?', content)
        if age_match:
            min_age = int(age_match.group(1))
            max_age = int(age_match.group(2)) if age_match.group(2) else min_age + 10
            filtered_params['relevant_demographics']['age_range'] = (min_age, max_age)
        elif 'elderly' in content or 'geriatric' in content:
            filtered_params['relevant_demographics']['age_range'] = (65, 100)
        elif 'young' in content or 'pediatric' in content:
            filtered_params['relevant_demographics']['age_range'] = (0, 18)


def _extract_clinical_factors(content: str, filtered_params: Dict[str, Any]):
    """Extract clinical factors from document content."""
    for factor in CLINICAL_FACTORS:
        if factor in content and factor not in filtered_params['relevant_clinical_factors']:
            filtered_params['relevant_clinical_factors'].append(factor)

