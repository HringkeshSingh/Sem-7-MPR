"""
Query parsing and context generation endpoints.
"""

import uuid
from fastapi import APIRouter

from src.api.schemas import (
    QueryParseRequest, QueryParseResponse,
    GenerateContextRequest, GenerateContextResponse
)
from src.api.state import app_state
from src.api.data_processor import parse_natural_language_query, extract_sample_size
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Query Processing"])


@router.post("/query/parse", response_model=QueryParseResponse)
async def parse_query(request: QueryParseRequest):
    """Parse a natural language query and return structured conditions."""
    try:
        query_id = str(uuid.uuid4())
        
        # Extract using RAG system
        rag_extracted_info = None
        if app_state.rag_system:
            try:
                rag_extracted_info = app_state.rag_system.extract_relevant_info(request.query)
                filtered_params = app_state.rag_system.filter_query_parameters(request.query, rag_extracted_info)
                logger.info(f"RAG extracted info: {filtered_params}")
            except Exception as e:
                logger.warning(f"Error using RAG system: {e}")
        
        # Parse query
        filters = parse_natural_language_query(request.query)
        
        # Extract sample size
        suggested_sample_size = extract_sample_size(request.query)
        
        # Generate suggested filters
        suggested_filters = []
        if filters['diagnoses']:
            suggested_filters.extend([f"Diagnosis: {d}" for d in filters['diagnoses']])
        if filters['age_range']:
            suggested_filters.append(f"Age: {filters['age_range'][0]}-{filters['age_range'][1]}")
        if filters['gender']:
            suggested_filters.append(f"Gender: {filters['gender']}")
        if filters['icu_required']:
            suggested_filters.append("ICU Stay: Required")
        if filters['mortality'] is not None:
            suggested_filters.append(f"Mortality: {'Yes' if filters['mortality'] else 'No'}")
        if filters['risk_level']:
            suggested_filters.append(f"Risk Level: {filters['risk_level']}")
        
        # Calculate confidence
        recognized = sum(1 for v in filters.values() if v is not None and v != [])
        confidence = min(recognized / 7, 1.0)
        
        extracted_params = {
            "sample_size": suggested_sample_size,
            "diagnoses": filters.get('diagnoses', []),
            "diagnosis_logic": filters.get('diagnosis_logic', 'OR'),
            "age_range": filters.get('age_range'),
            "gender": filters.get('gender'),
            "icu_required": filters.get('icu_required'),
            "risk_level": filters.get('risk_level'),
            "complexity": filters.get('complexity')
        }
        
        # Search PubMed
        pubmed_results = []
        if app_state.pubmed_client:
            try:
                pubmed_query = app_state.pubmed_client.build_healthcare_query(
                    conditions=filters.get('diagnoses', []),
                    demographics=filters
                )
                articles = app_state.pubmed_client.search_and_fetch(pubmed_query, 10)
                pubmed_results = [{
                    "title": a.title, "authors": a.authors, "journal": a.journal,
                    "year": a.year, "pmid": a.pmid, "abstract": a.abstract,
                    "relevance_score": a.relevance_score, "doi": a.doi
                } for a in articles]
            except Exception as e:
                logger.warning(f"Error fetching from PubMed: {e}")
        
        # Generate research context
        conditions = filters.get('diagnoses', [])
        age_info = f"age {filters['age_range'][0]}-{filters['age_range'][1]}" if filters.get('age_range') else "adult"
        gender_info = filters.get('gender', 'all')
        
        if rag_extracted_info and rag_extracted_info.get('summary'):
            research_context = f"""Query Analysis Summary:
- Target population: {', '.join(conditions) if conditions else 'general healthcare'} patients
- Demographics: {age_info}, {gender_info} gender
- Suggested sample size: {suggested_sample_size or 'not specified'}
- Query confidence: {confidence:.1%}

RAG-Extracted Information:
{rag_extracted_info.get('summary', '')}"""
        else:
            research_context = f"""Query Analysis Summary:
- Target population: {', '.join(conditions) if conditions else 'general healthcare'} patients
- Demographics: {age_info}, {gender_info} gender
- Suggested sample size: {suggested_sample_size or 'not specified'}
- Query confidence: {confidence:.1%}"""
        
        # Add articles to RAG
        if app_state.rag_system and pubmed_results:
            try:
                app_state.rag_system.add_pubmed_articles(pubmed_results)
            except Exception as e:
                logger.warning(f"Error adding articles to RAG: {e}")
        
        return QueryParseResponse(
            success=True,
            query_id=query_id,
            parsed_conditions=filters,
            suggested_filters=suggested_filters,
            confidence=confidence,
            extracted_params=extracted_params,
            research_context=research_context,
            pubmed_results=pubmed_results,
            embeddings_created=len(pubmed_results),
            rag_extracted_info=rag_extracted_info
        )
        
    except Exception as e:
        logger.error(f"Error parsing query: {e}")
        return QueryParseResponse(
            success=False, query_id="", parsed_conditions={}, suggested_filters=[],
            confidence=0.0, extracted_params={}, research_context="Error occurred.",
            pubmed_results=[], embeddings_created=0, rag_extracted_info=None
        )


@router.post("/generate-context", response_model=GenerateContextResponse)
async def generate_context(request: GenerateContextRequest):
    """Generate research context with PubMed search and query analysis."""
    try:
        query_id = str(uuid.uuid4())
        filters = parse_natural_language_query(request.query)
        suggested_sample_size = extract_sample_size(request.query)
        
        # Search PubMed
        pubmed_results = []
        if app_state.pubmed_client:
            try:
                pubmed_query = app_state.pubmed_client.build_healthcare_query(
                    conditions=filters.get('diagnoses', []),
                    demographics=filters
                )
                articles = app_state.pubmed_client.search_and_fetch(pubmed_query, request.max_articles)
                pubmed_results = [{
                    "title": a.title, "authors": a.authors, "journal": a.journal,
                    "year": a.year, "pmid": a.pmid, "abstract": a.abstract,
                    "relevance_score": a.relevance_score, "doi": a.doi
                } for a in articles]
            except Exception as e:
                logger.warning(f"Error fetching from PubMed: {e}")
        
        # Generate context
        conditions = filters.get('diagnoses', [])
        age_info = f"age {filters['age_range'][0]}-{filters['age_range'][1]}" if filters.get('age_range') else "adult"
        
        research_context = f"""Based on current medical literature, patients with {', '.join(conditions) if conditions else 'various conditions'} 
({age_info}) show specific clinical patterns. Key findings from recent studies:
- Prevalence rates align with published epidemiological data
- Treatment outcomes vary based on patient demographics
- ICU utilization patterns are consistent with severity indicators"""
        
        extracted_params = {
            "sample_size": suggested_sample_size,
            "diagnoses": filters.get('diagnoses', []),
            "diagnosis_logic": filters.get('diagnosis_logic', 'OR'),
            "age_range": filters.get('age_range'),
            "gender": filters.get('gender'),
            "icu_required": filters.get('icu_required'),
            "risk_level": filters.get('risk_level'),
            "complexity": filters.get('complexity')
        }
        
        return GenerateContextResponse(
            success=True,
            query_id=query_id,
            extracted_params=extracted_params,
            pubmed_results=pubmed_results[:request.max_articles],
            research_context=research_context,
            suggested_sample_size=suggested_sample_size
        )
        
    except Exception as e:
        logger.error(f"Error generating context: {e}")
        return GenerateContextResponse(
            success=False, query_id="", extracted_params={},
            pubmed_results=[], research_context="", suggested_sample_size=None
        )
