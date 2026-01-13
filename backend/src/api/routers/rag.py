"""
RAG system endpoints.
"""

from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException

from config.settings import MODELS_DIR
from src.api.schemas import (
    RAGExtractRequest, AddDocumentsRequest,
    MultiSourceRetrieveRequest, CrossValidateRequest
)
from src.api.state import app_state
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["RAG System"])


# ==================== Basic RAG Endpoints ====================

@router.post("/rag/extract")
async def rag_extract_info(request: RAGExtractRequest):
    """Extract relevant information using RAG system."""
    if not app_state.rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        extracted_info = app_state.rag_system.extract_relevant_info(request.query, max_docs=request.max_docs)
        filtered_params = app_state.rag_system.filter_query_parameters(request.query, extracted_info)
        
        return {
            "success": True,
            "query": request.query,
            "extracted_info": extracted_info,
            "filtered_parameters": filtered_params,
            "summary": extracted_info.get('summary', 'No relevant information found.'),
            "confidence": extracted_info.get('confidence', 0.0),
            "num_documents": extracted_info.get('num_documents', 0)
        }
        
    except Exception as e:
        logger.error(f"Error in RAG extraction: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting information: {str(e)}")


@router.get("/rag/stats")
async def get_rag_stats():
    """Get RAG system statistics."""
    if not app_state.rag_system:
        return {"status": "not_initialized", "message": "RAG system not initialized"}
    
    return {"status": "initialized", "stats": app_state.rag_system.get_vectorstore_stats()}


@router.post("/rag/add-documents")
async def add_documents_to_rag(request: AddDocumentsRequest):
    """Add documents to the RAG system."""
    if not app_state.rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    metadatas = [request.metadata] * len(request.texts) if request.metadata else None
    app_state.rag_system.add_texts(request.texts, metadatas)
    
    return {
        "success": True,
        "message": f"Added {len(request.texts)} documents to RAG system",
        "stats": app_state.rag_system.get_vectorstore_stats()
    }


# ==================== Enhanced RAG Endpoints ====================

@router.post("/enhanced-rag/retrieve")
async def enhanced_retrieve(request: MultiSourceRetrieveRequest):
    """Retrieve from multiple sources (PubMed, ClinicalTrials.gov, WHO, etc.)"""
    try:
        enhanced_rag = app_state.get_enhanced_rag()
        
        result = enhanced_rag.retrieve_and_extract(
            query=request.query,
            max_results_per_source=request.max_results_per_source,
            use_cache=request.use_cache,
            index_results=request.index_results,
            sources=request.sources
        )
        
        return {"success": True, **result}
        
    except Exception as e:
        logger.error(f"Error in enhanced retrieval: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving: {str(e)}")


@router.post("/enhanced-rag/validate")
async def cross_validate_data(request: CrossValidateRequest):
    """Cross-validate generated data against medical literature."""
    try:
        enhanced_rag = app_state.get_enhanced_rag()
        report = enhanced_rag.validate_generated_data(
            generated_data=request.generated_data,
            query=request.query
        )
        
        return {"success": True, "validation": report.to_dict()}
        
    except Exception as e:
        logger.error(f"Error in cross-validation: {e}")
        raise HTTPException(status_code=500, detail=f"Error validating: {str(e)}")


@router.get("/enhanced-rag/sources")
async def get_available_sources():
    """Get list of available data sources."""
    try:
        enhanced_rag = app_state.get_enhanced_rag()
        
        return {
            "success": True,
            "available_sources": enhanced_rag.get_available_sources(),
            "system_stats": enhanced_rag.get_system_stats()
        }
        
    except Exception as e:
        logger.error(f"Error getting sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enhanced-rag/stats")
async def get_enhanced_rag_stats():
    """Get enhanced RAG system statistics."""
    if app_state.enhanced_rag is None:
        return {"success": True, "status": "not_initialized", "message": "Enhanced RAG not yet initialized"}
    
    return {"success": True, "status": "initialized", "stats": app_state.enhanced_rag.get_system_stats()}


@router.post("/enhanced-rag/add-custom")
async def add_custom_documents(documents: List[Dict[str, Any]], source_name: str = "custom"):
    """Add custom documents to the enhanced RAG knowledge base."""
    try:
        enhanced_rag = app_state.get_enhanced_rag()
        count = enhanced_rag.add_custom_documents(documents, source_name)
        
        return {"success": True, "message": f"Added {count} document chunks", "source": source_name}
        
    except Exception as e:
        logger.error(f"Error adding custom documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/enhanced-rag/cache")
async def clear_enhanced_rag_cache():
    """Clear the enhanced RAG document cache."""
    if app_state.enhanced_rag:
        app_state.enhanced_rag.clear_cache()
    
    return {"success": True, "message": "Cache cleared"}
