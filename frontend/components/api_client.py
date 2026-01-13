"""
Enhanced API Client for Healthcare Data Generation System.
Supports RAG-augmented generation, query expansion, validation, and more.
"""

import requests
import time
from typing import Dict, Optional, Any, List


class APIClient:
    """Client for interacting with the Healthcare Data Generation API."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Healthcare-Frontend/2.0'
        })
    
    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                 timeout: int = 30) -> Dict[str, Any]:
        """Make an API request with error handling."""
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=timeout)
            else:
                response = self.session.get(url, timeout=timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}"
                }
                
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timeout"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Connection error - API not available"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    # ==================== Health & Status ====================
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health with retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(f"{self.base_url}/health", timeout=10)
                if response.status_code == 200:
                    return response.json()
                return {"status": "error", "message": f"HTTP {response.status_code}"}
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return {"status": "error", "message": str(e)}
        return {"status": "error", "message": "Max retries exceeded"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get detailed system status."""
        return self._request("GET", "/system/status", timeout=10)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self._request("GET", "/statistics", timeout=10)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self._request("GET", "/model/info", timeout=10)
    
    def get_examples(self) -> Dict[str, Any]:
        """Get example queries."""
        result = self._request("GET", "/examples", timeout=10)
        if "examples" not in result:
            result["examples"] = []
        return result
    
    # ==================== Basic Generation ====================
    
    def generate_data(self, query: str, num_patients: int = 100, 
                     format_type: str = "json") -> Dict[str, Any]:
        """Generate synthetic healthcare data."""
        return self._request("POST", "/generate", {
            "query": query,
            "num_patients": num_patients,
            "format": format_type
        })
    
    def parse_query(self, query: str, max_articles: int = 50) -> Dict[str, Any]:
        """Parse a natural language query with RAG enhancement."""
        return self._request("POST", "/query/parse", {
            "query": query,
            "max_articles": max_articles
        })
    
    def generate_context(self, query: str, max_articles: int = 50) -> Dict[str, Any]:
        """Generate research context for a query."""
        return self._request("POST", "/generate-context", {
            "query": query,
            "max_articles": max_articles
        })
    
    # ==================== RAG System ====================
    
    def rag_extract(self, query: str, max_docs: Optional[int] = None) -> Dict[str, Any]:
        """Extract relevant information using RAG system."""
        payload = {"query": query}
        if max_docs:
            payload["max_docs"] = max_docs
        return self._request("POST", "/rag/extract", payload)
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        return self._request("GET", "/rag/stats", timeout=10)
    
    def add_documents(self, texts: List[str], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Add documents to the RAG system."""
        return self._request("POST", "/rag/add-documents", {
            "texts": texts,
            "metadata": metadata
        })
    
    # ==================== RAG-Enhanced Generation ====================
    
    def rag_generate(self, query: str, num_patients: int = 100,
                    use_multi_hop: bool = True, include_citations: bool = True,
                    include_validation: bool = False) -> Dict[str, Any]:
        """Generate synthetic data using RAG-augmented context."""
        return self._request("POST", "/rag-generate/generate", {
            "query": query,
            "num_patients": num_patients,
            "use_multi_hop": use_multi_hop,
            "include_citations": include_citations,
            "include_validation": include_validation
        }, timeout=60)
    
    def expand_query(self, query: str, include_icd10: bool = True,
                    max_expansions: int = 10) -> Dict[str, Any]:
        """Expand a query with semantic understanding."""
        return self._request("POST", "/rag-generate/expand-query", {
            "query": query,
            "include_icd10": include_icd10,
            "max_expansions": max_expansions
        })
    
    def validate_data(self, data: List[Dict], validation_types: Optional[List[str]] = None,
                     query_context: Optional[str] = None) -> Dict[str, Any]:
        """Validate generated data using multiple validators."""
        return self._request("POST", "/rag-generate/validate", {
            "data": data,
            "validation_types": validation_types,
            "query_context": query_context
        })
    
    def get_rag_generate_examples(self) -> Dict[str, Any]:
        """Get example queries for RAG-augmented generation."""
        return self._request("GET", "/rag-generate/examples", timeout=10)
    
    def get_rag_generate_stats(self) -> Dict[str, Any]:
        """Get RAG-augmented generation statistics."""
        return self._request("GET", "/rag-generate/stats", timeout=10)
    
    # ==================== Enhanced RAG (Multi-Source) ====================
    
    def enhanced_retrieve(self, query: str, max_results: int = 10,
                         sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """Retrieve from multiple sources using enhanced RAG."""
        return self._request("POST", "/enhanced-rag/retrieve", {
            "query": query,
            "max_results_per_source": max_results,
            "sources": sources,
            "use_cache": True,
            "index_results": True
        })
    
    def cross_validate(self, generated_data: Dict, query: Optional[str] = None) -> Dict[str, Any]:
        """Cross-validate generated data against literature."""
        return self._request("POST", "/enhanced-rag/validate", {
            "generated_data": generated_data,
            "query": query
        })
    
    def get_available_sources(self) -> Dict[str, Any]:
        """Get available data sources for enhanced RAG."""
        return self._request("GET", "/enhanced-rag/sources", timeout=10)
    
    def get_enhanced_rag_stats(self) -> Dict[str, Any]:
        """Get enhanced RAG system statistics."""
        return self._request("GET", "/enhanced-rag/stats", timeout=10)
    
    # ==================== Utilities ====================
    
    def test_endpoint(self, endpoint: str, method: str = "GET", 
                     data: Optional[Dict] = None) -> Dict[str, Any]:
        """Test a specific API endpoint."""
        start_time = time.time()
        result = self._request(method, endpoint, data)
        result["response_time_ms"] = (time.time() - start_time) * 1000
        return result
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get API information and available endpoints."""
        return {
            "base_url": self.base_url,
            "version": "2.0",
            "description": "Healthcare Data Generation API with RAG Enhancement",
            "endpoints": [
                {"path": "/health", "method": "GET", "description": "Health check"},
                {"path": "/statistics", "method": "GET", "description": "Dataset statistics"},
                {"path": "/generate", "method": "POST", "description": "Generate data"},
                {"path": "/query/parse", "method": "POST", "description": "Parse query with RAG"},
                {"path": "/rag/extract", "method": "POST", "description": "RAG extraction"},
                {"path": "/rag-generate/generate", "method": "POST", "description": "RAG-augmented generation"},
                {"path": "/rag-generate/expand-query", "method": "POST", "description": "Semantic query expansion"},
                {"path": "/rag-generate/validate", "method": "POST", "description": "Multi-validator validation"},
                {"path": "/enhanced-rag/retrieve", "method": "POST", "description": "Multi-source retrieval"},
            ]
        }
