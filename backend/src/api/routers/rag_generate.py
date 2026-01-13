"""
RAG-Enhanced Data Generation Endpoints.

Provides endpoints for:
- RAG-augmented synthetic data generation
- Query expansion with semantic understanding
- Evidence-based validation
- Multi-hop reasoning for complex scenarios
"""

import tempfile
from datetime import datetime
from typing import Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from src.api.schemas import (
    RAGGenerateRequest, RAGGenerateResponse,
    ExpandQueryRequest, ExpandQueryResponse,
    ValidateDataRequest, ValidationReportResponse,
    EvidenceCitationResponse, ReasoningStepResponse
)
from src.api.state import app_state
from src.api.query_parser import EnhancedQueryParser
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/rag-generate", tags=["RAG-Enhanced Generation"])

# Initialize parser
_parser = EnhancedQueryParser()


def _get_rag_generator():
    """Get or create RAG data generator."""
    if not hasattr(app_state, 'rag_generator') or app_state.rag_generator is None:
        try:
            from src.api.rag_data_generator import RAGDataGenerator
            
            app_state.rag_generator = RAGDataGenerator(
                ctgan_model=app_state.ctgan_model,
                rag_system=app_state.rag_system,
                enhanced_rag=app_state.enhanced_rag,
                original_dataset=app_state.original_dataset
            )
        except Exception as e:
            logger.error(f"Failed to initialize RAG generator: {e}")
            return None
    
    return app_state.rag_generator


@router.post("/generate", response_model=RAGGenerateResponse)
async def rag_generate_data(request: RAGGenerateRequest):
    """
    Generate synthetic healthcare data using RAG-augmented context.
    
    This endpoint:
    1. Parses the query to extract medical entities
    2. Retrieves relevant medical literature
    3. Applies literature-derived constraints
    4. Generates synthetic data with evidence support
    5. Optionally validates against literature
    """
    try:
        generator = _get_rag_generator()
        
        if generator is None:
            raise HTTPException(
                status_code=503,
                detail="RAG generator not available. Check model and RAG system initialization."
            )
        
        logger.info(f"RAG-generating {request.num_patients} patients for: '{request.query}'")
        
        # Generate with RAG
        if request.include_validation:
            result, validation = generator.generate_with_validation(
                query=request.query,
                num_samples=request.num_patients,
                validate=True
            )
        else:
            result = generator.generate(
                query=request.query,
                num_samples=request.num_patients,
                use_multi_hop=request.use_multi_hop,
                include_citations=request.include_citations
            )
            validation = None
        
        # Convert citations
        citations = [
            EvidenceCitationResponse(
                source=c.source,
                source_type=c.source_type,
                title=c.title,
                url=c.url,
                relevance_score=c.relevance_score,
                excerpt=c.excerpt
            )
            for c in result.citations
        ]
        
        # Convert reasoning chain
        reasoning = [
            ReasoningStepResponse(
                step_number=r.step_number,
                description=r.description,
                query=r.query,
                result_summary=r.result_summary,
                confidence=r.confidence,
                sources=r.sources
            )
            for r in result.reasoning_chain
        ]
        
        # Prepare response data
        response_data = None
        if request.output_format == "json":
            response_data = result.data.to_dict('records')
        
        return RAGGenerateResponse(
            success=True,
            message=f"Generated {result.num_records} patients with RAG augmentation",
            num_records=result.num_records,
            confidence_score=result.confidence_score,
            data=response_data,
            citations=citations,
            reasoning_chain=reasoning,
            applied_constraints=result.applied_constraints,
            generation_time_ms=result.generation_time_ms,
            validation_report=validation,
            metadata={
                "query": request.query,
                "use_multi_hop": request.use_multi_hop,
                "include_citations": request.include_citations,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in RAG generation: {e}")
        raise HTTPException(status_code=500, detail=f"RAG generation error: {str(e)}")


@router.get("/generate/csv")
async def rag_generate_csv(
    query: str = Query(..., description="Natural language query"),
    num_patients: int = Query(100, ge=1, le=10000),
    use_multi_hop: bool = Query(True, description="Use multi-hop reasoning")
):
    """Generate RAG-augmented synthetic data and return as CSV file."""
    try:
        generator = _get_rag_generator()
        
        if generator is None:
            raise HTTPException(status_code=503, detail="RAG generator not available")
        
        result = generator.generate(
            query=query,
            num_samples=num_patients,
            use_multi_hop=use_multi_hop,
            include_citations=False  # Skip citations for CSV
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            result.data.to_csv(f.name, index=False)
            temp_path = f.name
        
        return FileResponse(
            temp_path,
            media_type='text/csv',
            filename=f"rag_synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/expand-query", response_model=ExpandQueryResponse)
async def expand_query(request: ExpandQueryRequest):
    """
    Expand a query with semantic understanding and medical terminology.
    
    Returns:
    - Normalized query
    - Extracted medical entities with ICD-10 codes
    - Related terms and concepts
    - Suggested alternative queries
    """
    try:
        # Set up parser with available systems
        if app_state.rag_system:
            _parser.set_rag_system(app_state.rag_system)
        
        # Expand query
        expanded = _parser.expand_query(request.query)
        
        # Get ICD-10 codes if requested
        icd10_codes = []
        if request.include_icd10:
            icd10_codes = _parser.get_icd10_codes(request.query)
        
        # Get suggested queries
        suggested = _parser.suggest_related_queries(request.query, max_suggestions=5)
        
        return ExpandQueryResponse(
            success=True,
            original_query=expanded.original_query,
            normalized_query=expanded.normalized_query,
            entities=[e.to_dict() for e in expanded.entities],
            expansions=expanded.expansions[:request.max_expansions],
            semantic_concepts=expanded.semantic_concepts,
            icd10_codes=icd10_codes,
            filters=expanded.filters,
            confidence=expanded.confidence,
            suggested_queries=suggested
        )
        
    except Exception as e:
        logger.error(f"Error expanding query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=ValidationReportResponse)
async def validate_generated_data(request: ValidateDataRequest):
    """
    Validate generated data using multiple validators.
    
    Validation types:
    - clinical: Clinical validity checks
    - literature: Comparison with epidemiological data
    - temporal: Current medical knowledge checks
    - confidence: Confidence scoring
    """
    try:
        from src.validation import (
            ClinicalValidator,
            LiteratureValidator,
            TemporalValidator,
            ConfidenceScorer
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        if len(df) == 0:
            return ValidationReportResponse(
                success=False,
                overall_valid=False,
                overall_score=0.0,
                confidence=0.0,
                validation_results={"error": "No data provided"},
                issues_summary={},
                recommendations=["Provide data to validate"]
            )
        
        # Determine validation types
        validation_types = request.validation_types or ['clinical', 'temporal', 'confidence']
        
        results = {}
        all_issues = {}
        all_recommendations = []
        scores = []
        
        # Clinical validation
        if 'clinical' in validation_types:
            try:
                clinical = ClinicalValidator()
                
                if request.query_context and app_state.rag_system:
                    rag_result = app_state.rag_system.extract_relevant_info(request.query_context)
                    docs = rag_result.get('relevant_info', [])
                    clinical_report = clinical.validate_with_context(
                        df,
                        retrieved_documents=docs,
                        query_context={'conditions': _parser.parse_query(request.query_context).get('diagnoses', [])}
                    )
                else:
                    clinical_report = clinical.validate(df)
                
                results['clinical'] = clinical_report.to_dict()
                scores.append(clinical_report.overall_score)
                all_issues.update(clinical_report.issues_by_severity)
                all_recommendations.extend(clinical_report.recommendations)
            except Exception as e:
                results['clinical'] = {"error": str(e)}
        
        # Literature validation
        if 'literature' in validation_types:
            try:
                literature = LiteratureValidator()
                
                if app_state.rag_system and request.query_context:
                    rag_result = app_state.rag_system.extract_relevant_info(request.query_context)
                    docs = rag_result.get('relevant_info', [])
                    lit_report = literature.validate_with_literature(df, docs)
                else:
                    lit_report = literature.validate(df)
                
                results['literature'] = lit_report.to_dict()
                scores.append(lit_report.overall_alignment)
                all_recommendations.append(lit_report.summary)
            except Exception as e:
                results['literature'] = {"error": str(e)}
        
        # Temporal validation
        if 'temporal' in validation_types:
            try:
                temporal = TemporalValidator()
                temporal_report = temporal.validate(df)
                
                results['temporal'] = temporal_report.to_dict()
                scores.append(temporal_report.currency_score)
                all_recommendations.extend(temporal_report.recommendations)
            except Exception as e:
                results['temporal'] = {"error": str(e)}
        
        # Confidence scoring
        if 'confidence' in validation_types:
            try:
                scorer = ConfidenceScorer()
                
                # Use validation results if available
                confidence_report = scorer.score_with_validation(
                    df,
                    clinical_report=results.get('clinical'),
                    literature_report=results.get('literature'),
                    temporal_report=results.get('temporal')
                )
                
                results['confidence'] = confidence_report.to_dict()
                all_recommendations.extend(confidence_report.recommendations)
            except Exception as e:
                results['confidence'] = {"error": str(e)}
        
        # Calculate overall
        overall_score = sum(scores) / len(scores) if scores else 0.5
        overall_valid = overall_score >= 0.7
        
        return ValidationReportResponse(
            success=True,
            overall_valid=overall_valid,
            overall_score=overall_score,
            confidence=results.get('confidence', {}).get('overall_confidence', overall_score),
            validation_results=results,
            issues_summary=all_issues,
            recommendations=list(set(all_recommendations))[:10]
        )
        
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/examples")
async def get_example_queries():
    """Get example queries for RAG-augmented generation."""
    examples = _parser.generate_example_queries()
    
    return {
        "examples": examples,
        "tips": [
            "Include specific conditions for better results (e.g., 'diabetes', 'hypertension')",
            "Specify age groups like 'elderly', 'young adult', or exact ranges",
            "Add clinical context like 'ICU', 'emergency', 'critical'",
            "Mention complications for more realistic data",
            "Use 'and' for patients with multiple conditions"
        ],
        "supported_conditions": list(MEDICAL_TERMINOLOGY.keys()) if 'MEDICAL_TERMINOLOGY' in dir() else [
            "DIABETES", "CARDIOVASCULAR", "HYPERTENSION", "RENAL", 
            "RESPIRATORY", "SEPSIS", "NEUROLOGICAL", "TRAUMA", "CANCER"
        ]
    }


@router.get("/stats")
async def get_rag_generation_stats():
    """Get statistics about RAG-augmented generation."""
    generator = _get_rag_generator()
    
    stats = {
        "generator_available": generator is not None,
        "ctgan_model_loaded": app_state.ctgan_model is not None,
        "rag_system_available": app_state.rag_system is not None,
        "enhanced_rag_available": getattr(app_state, 'enhanced_rag', None) is not None,
        "original_dataset_size": len(app_state.original_dataset) if app_state.original_dataset is not None else 0
    }
    
    if generator:
        stats["generation_history_count"] = len(generator.get_generation_history())
    
    if app_state.rag_system:
        try:
            rag_stats = app_state.rag_system.get_stats()
            stats["rag_stats"] = rag_stats
        except:
            pass
    
    return stats


# Import medical terminology for examples endpoint
try:
    from src.api.query_parser import MEDICAL_TERMINOLOGY
except:
    MEDICAL_TERMINOLOGY = {}
