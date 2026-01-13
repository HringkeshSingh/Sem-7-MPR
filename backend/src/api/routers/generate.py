"""
Data generation endpoints.
"""

import tempfile
from datetime import datetime
from typing import Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from config.settings import API_CONFIG
from src.api.schemas import (
    GenerateRequest, GenerateResponse, 
    DataGenerateRequest, DataGenerateResponse
)
from src.api.state import app_state
from src.api.data_processor import (
    parse_natural_language_query, 
    apply_filters_to_data, 
    generate_synthetic_data
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Data Generation"])


@router.post("/generate", response_model=GenerateResponse)
async def generate_data(request: GenerateRequest):
    """Generate synthetic healthcare data based on natural language query."""
    try:
        logger.info(f"Generating data for query: '{request.query}' ({request.num_patients} patients)")
        
        filters = parse_natural_language_query(request.query)
        synthetic_data = generate_synthetic_data(request.num_patients, filters)
        
        original_matches = None
        if request.include_original and app_state.original_dataset is not None:
            filtered_original = apply_filters_to_data(app_state.original_dataset, filters)
            original_matches = filtered_original.head(50)
        
        response_data = synthetic_data.to_dict('records')
        
        metadata = {
            'generated_count': len(synthetic_data),
            'generation_timestamp': datetime.now().isoformat(),
            'filters_applied': {k: v for k, v in filters.items() if v is not None and v != []},
            'original_matches_count': len(original_matches) if original_matches is not None else 0
        }
        
        if original_matches is not None and len(original_matches) > 0:
            metadata['original_matches'] = original_matches.to_dict('records')
        
        query_info = {
            'original_query': request.query,
            'parsed_conditions': [k for k, v in filters.items() if v is not None and v != []],
            'num_requested': request.num_patients,
            'num_generated': len(synthetic_data)
        }
        
        return GenerateResponse(
            success=True,
            message=f"Successfully generated {len(synthetic_data)} patients",
            data=response_data,
            metadata=metadata,
            query_info=query_info
        )
        
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating data: {str(e)}")


@router.get("/generate/csv")
async def generate_data_csv(
    query: str = Query(..., description="Natural language query"),
    num_patients: int = Query(100, ge=1, le=API_CONFIG['max_generated_rows']),
):
    """Generate synthetic data and return as CSV file."""
    try:
        filters = parse_natural_language_query(query)
        synthetic_data = generate_synthetic_data(num_patients, filters)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            synthetic_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        return FileResponse(
            temp_path,
            media_type='text/csv',
            filename=f"synthetic_healthcare_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
    except Exception as e:
        logger.error(f"Error generating CSV data: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating CSV data: {str(e)}")


@router.post("/data/generate", response_model=DataGenerateResponse)
async def generate_data_endpoint(request: DataGenerateRequest):
    """Generate synthetic data based on query parameters."""
    try:
        import uuid
        task_id = str(uuid.uuid4())
        
        sample_size = request.sample_size or 100
        
        filters = {
            'diagnoses': request.diagnoses or [],
            'diagnosis_logic': request.diagnosis_logic or 'OR',
            'age_range': request.age_range,
            'gender': request.gender,
            'icu_required': request.icu_required,
            'mortality': request.mortality,
            'risk_level': request.risk_level,
            'complexity': request.complexity
        }
        
        synthetic_data = generate_synthetic_data(sample_size, filters)
        
        app_state.generation_results[task_id] = {
            "status": "completed",
            "data": synthetic_data.to_dict('records'),
            "metadata": {
                "generated_count": len(synthetic_data),
                "total_patients": len(synthetic_data),
                "generation_timestamp": datetime.now().isoformat(),
                "query_id": request.query_id,
                "sample_size": sample_size,
                "mortality_rate": synthetic_data.get('mortality', pd.Series([0])).mean(),
                "avg_age": synthetic_data.get('age', pd.Series([0])).mean(),
                "icu_rate": synthetic_data.get('has_icu_stay', pd.Series([0])).mean(),
                "gender_distribution": synthetic_data.get('gender', pd.Series()).value_counts().to_dict(),
            }
        }
        
        return DataGenerateResponse(
            success=True,
            task_id=task_id,
            message=f"Data generation completed for {len(synthetic_data)} patients"
        )
        
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        return DataGenerateResponse(success=False, task_id="", message=f"Error: {str(e)}")


@router.get("/data/status/{task_id}")
async def get_generation_status(task_id: str):
    """Get the status of a data generation task."""
    if task_id not in app_state.generation_results:
        return {"status": "not_found", "message": "Task not found"}
    
    result = app_state.generation_results[task_id]
    return {
        "status": result["status"],
        "progress": 100 if result["status"] == "completed" else 0,
        "message": f"Generated {result['metadata']['generated_count']} patients"
    }


@router.get("/data/download/{task_id}")
async def download_data(task_id: str):
    """Download generated data."""
    if task_id not in app_state.generation_results:
        return {"success": False, "error": "Task not found"}
    
    result = app_state.generation_results[task_id]
    return {
        "success": True,
        "data": result["data"],
        "metadata": result["metadata"]
    }
