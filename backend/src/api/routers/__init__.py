"""
API Routers for modular endpoint organization.
"""

from .health import router as health_router
from .generate import router as generate_router
from .query import router as query_router
from .rag import router as rag_router
from .rag_generate import router as rag_generate_router

__all__ = [
    'health_router', 
    'generate_router', 
    'query_router', 
    'rag_router',
    'rag_generate_router'
]
