"""
Pydantic models for API requests and responses.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from config.settings import API_CONFIG


# ==================== Request Models ====================

class GenerateRequest(BaseModel):
    query: str = Field(..., description="Natural language query for data generation")
    num_patients: int = Field(default=100, ge=1, le=API_CONFIG['max_generated_rows'])
    include_original: bool = Field(default=False, description="Include matching original data")
    format: str = Field(default="json", pattern="^(json|csv)$")  # type: ignore
    
    @validator('query')
    def validate_query(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Query must be at least 3 characters long')
        return v.strip()


class QueryParseRequest(BaseModel):
    query: str = Field(..., description="Natural language query to parse")


class GenerateContextRequest(BaseModel):
    query: str = Field(..., description="Query for generating research context")
    max_articles: int = Field(default=50, ge=1, le=100)


class DataGenerateRequest(BaseModel):
    query_id: str
    sample_size: Optional[int] = None
    age_range: Optional[List[int]] = None
    use_research_context: bool = True
    gender: Optional[str] = None
    diagnoses: Optional[List[str]] = None
    diagnosis_logic: Optional[str] = 'OR'
    icu_required: Optional[bool] = None
    mortality: Optional[bool] = None
    risk_level: Optional[str] = None
    complexity: Optional[str] = None


class RAGExtractRequest(BaseModel):
    query: str = Field(..., description="User query to extract relevant information from")
    max_docs: Optional[int] = Field(default=None, ge=1, le=20)


class AddDocumentsRequest(BaseModel):
    texts: List[str] = Field(..., description="List of text documents to add")
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class MultiSourceRetrieveRequest(BaseModel):
    query: str = Field(..., description="Search query for multi-source retrieval")
    max_results_per_source: int = Field(default=10, ge=1, le=50)
    sources: Optional[List[str]] = Field(default=None)
    use_cache: bool = Field(default=True)
    index_results: bool = Field(default=True)


class CrossValidateRequest(BaseModel):
    generated_data: Dict[str, Any] = Field(..., description="Generated data to validate")
    query: Optional[str] = Field(default=None)


class RAGGenerateRequest(BaseModel):
    """Request for RAG-augmented data generation."""
    query: str = Field(..., description="Natural language query for data generation")
    num_patients: int = Field(default=100, ge=1, le=10000)
    use_multi_hop: bool = Field(default=True, description="Use multi-hop reasoning")
    include_citations: bool = Field(default=True, description="Include evidence citations")
    include_validation: bool = Field(default=False, description="Validate against literature")
    output_format: str = Field(default="json", pattern="^(json|csv)$")
    
    @validator('query')
    def validate_query(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Query must be at least 3 characters long')
        return v.strip()


class ExpandQueryRequest(BaseModel):
    """Request for query expansion with semantic understanding."""
    query: str = Field(..., description="Query to expand")
    include_icd10: bool = Field(default=True, description="Include ICD-10 codes")
    max_expansions: int = Field(default=10, ge=1, le=50)


class ValidateDataRequest(BaseModel):
    """Request for comprehensive data validation."""
    data: List[Dict[str, Any]] = Field(..., description="Data records to validate")
    validation_types: Optional[List[str]] = Field(
        default=None,
        description="Types: clinical, literature, temporal, confidence"
    )
    query_context: Optional[str] = Field(default=None)


# ==================== Response Models ====================

class GenerateResponse(BaseModel):
    success: bool
    message: str
    data: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any]
    query_info: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    dataset_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_type: str
    training_timestamp: str
    training_samples: int
    training_features: int
    model_parameters: Dict[str, Any]


class QueryParseResponse(BaseModel):
    success: bool
    query_id: str
    parsed_conditions: Dict[str, Any]
    suggested_filters: List[str]
    confidence: float
    extracted_params: Dict[str, Any]
    research_context: str
    pubmed_results: List[Dict[str, Any]]
    embeddings_created: int
    rag_extracted_info: Optional[Dict[str, Any]] = None


class GenerateContextResponse(BaseModel):
    success: bool
    query_id: str
    extracted_params: Dict[str, Any]
    pubmed_results: List[Dict[str, Any]]
    research_context: str
    suggested_sample_size: Optional[int] = None


class DataGenerateResponse(BaseModel):
    success: bool
    task_id: str
    message: str


class EvidenceCitationResponse(BaseModel):
    """Evidence citation in response."""
    source: str
    source_type: str
    title: Optional[str] = None
    url: Optional[str] = None
    relevance_score: float = 0.0
    excerpt: Optional[str] = None


class ReasoningStepResponse(BaseModel):
    """Reasoning step in multi-hop reasoning."""
    step_number: int
    description: str
    query: str
    result_summary: str
    confidence: float
    sources: List[str] = []


class RAGGenerateResponse(BaseModel):
    """Response for RAG-augmented data generation."""
    success: bool
    message: str
    num_records: int
    confidence_score: float
    data: Optional[List[Dict[str, Any]]] = None
    citations: List[EvidenceCitationResponse] = []
    reasoning_chain: List[ReasoningStepResponse] = []
    applied_constraints: List[Dict[str, Any]] = []
    generation_time_ms: float = 0.0
    validation_report: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}


class ExpandQueryResponse(BaseModel):
    """Response for query expansion."""
    success: bool
    original_query: str
    normalized_query: str
    entities: List[Dict[str, Any]] = []
    expansions: List[str] = []
    semantic_concepts: List[str] = []
    icd10_codes: List[str] = []
    filters: Dict[str, Any] = {}
    confidence: float = 0.0
    suggested_queries: List[str] = []


class ValidationReportResponse(BaseModel):
    """Response for data validation."""
    success: bool
    overall_valid: bool
    overall_score: float
    confidence: float
    validation_results: Dict[str, Any] = {}
    issues_summary: Dict[str, int] = {}
    recommendations: List[str] = []
