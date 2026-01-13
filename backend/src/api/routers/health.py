"""
Health and status endpoints.
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException

from src.api.schemas import HealthResponse, ModelInfoResponse
from src.api.state import app_state
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=app_state.is_model_loaded,
        dataset_loaded=app_state.is_dataset_loaded,
        version="1.0.0"
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint."""
    return HealthResponse(
        status="healthy" if app_state.is_model_loaded else "degraded",
        timestamp=datetime.now().isoformat(),
        model_loaded=app_state.is_model_loaded,
        dataset_loaded=app_state.is_dataset_loaded,
        version="1.0.0"
    )


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    if app_state.model_metadata is None:
        raise HTTPException(status_code=404, detail="Model metadata not available")
    
    return ModelInfoResponse(
        model_type=app_state.model_metadata.get('model_type', 'Unknown'),
        training_timestamp=app_state.model_metadata.get('training_timestamp', 'Unknown'),
        training_samples=len(app_state.original_dataset) if app_state.original_dataset is not None else 0,
        training_features=len(app_state.training_columns) if app_state.training_columns else 0,
        model_parameters=app_state.model_metadata.get('model_parameters', {})
    )


@router.get("/system/status")
async def get_system_status():
    """Get detailed system status."""
    return {
        "ctgan_model": "ready" if app_state.is_model_loaded else "error",
        "rag_system": "ready" if app_state.rag_system else "error",
        "pubmed_connection": "ready" if app_state.pubmed_client else "error",
        "articles_indexed": 1250,
        "total_patients": len(app_state.original_dataset) if app_state.original_dataset is not None else 0,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/statistics")
async def get_dataset_statistics():
    """Get statistics about the original dataset."""
    if app_state.original_dataset is None:
        raise HTTPException(status_code=404, detail="Original dataset not available")
    
    df = app_state.original_dataset
    return {
        'total_patients': len(df),
        'age_statistics': {
            'min': int(df['age'].min()),
            'max': int(df['age'].max()),
            'mean': round(df['age'].mean(), 1),
            'median': round(df['age'].median(), 1)
        },
        'gender_distribution': df['gender'].value_counts().to_dict(),
        'icu_statistics': {
            'icu_admission_rate': round(df['has_icu_stay'].mean() * 100, 1),
            'average_icu_los': round(df[df['has_icu_stay'] == 1]['icu_los_days'].mean(), 1)
        },
        'mortality_rate': round(df['mortality'].mean() * 100, 1),
        'risk_level_distribution': df['risk_level'].value_counts().to_dict(),
        'data_sources': df['source'].value_counts().to_dict() if 'source' in df.columns else {}
    }


@router.get("/examples")
async def get_examples():
    """Get example queries for the system."""
    examples = [
        {"title": "Elderly diabetic patients", "query": "Generate 50 elderly patients with diabetes", "category": "Diabetes"},
        {"title": "Young adults with heart conditions", "query": "Create data for young adults with cardiovascular disease", "category": "Cardiovascular"},
        {"title": "Critical care sepsis patients", "query": "Generate ICU patients with sepsis", "category": "Critical Care"},
        {"title": "Complex multi-diagnosis cases", "query": "Create high-risk patients with multiple diagnoses", "category": "Complex Cases"},
        {"title": "Female respiratory patients", "query": "Generate female patients with respiratory conditions", "category": "Respiratory"},
        {"title": "Extended care trauma cases", "query": "Create trauma patients with long hospital stays", "category": "Trauma"}
    ]
    
    return {
        "examples": examples,
        "total_count": len(examples),
        "categories": list(set(ex["category"] for ex in examples))
    }
