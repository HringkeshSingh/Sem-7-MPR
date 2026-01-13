"""
FastAPI application for healthcare data generation.

This is the main entry point for the API. All endpoint logic has been
modularized into separate routers for better maintainability.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from config.settings import API_CONFIG
from src.utils.logging_config import setup_logging, get_logger
from src.api.state import app_state
from src.api.routers import health_router, generate_router, query_router, rag_router, rag_generate_router

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare Data Generation API",
    description="Generate synthetic healthcare data with multi-diagnosis support for research",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(generate_router)
app.include_router(query_router)
app.include_router(rag_router)
app.include_router(rag_generate_router)


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Healthcare Data Generation API")
    try:
        app_state.load_model_and_data()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise


if __name__ == "__main__":
    uvicorn.run(app, host=API_CONFIG['host'], port=API_CONFIG['port'])
