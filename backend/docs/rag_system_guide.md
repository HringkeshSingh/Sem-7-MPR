# RAG System Guide

## Overview

The RAG (Retrieval-Augmented Generation) system uses LangChain to extract only relevant information from user prompts. It processes healthcare documents, creates embeddings, and retrieves the most relevant information based on semantic similarity.

## Features

- **Document Processing**: Automatically processes and chunks healthcare documents
- **Semantic Search**: Uses embeddings to find relevant information based on meaning, not just keywords
- **PubMed Integration**: Automatically adds PubMed articles to the knowledge base
- **Relevance Filtering**: Only returns information above a similarity threshold
- **Query Parameter Extraction**: Automatically extracts relevant conditions, demographics, and clinical factors

## API Endpoints

### 1. Extract Relevant Information

**POST** `/rag/extract`

Extract only relevant information from a user query.

**Request Body:**
```json
{
  "query": "elderly patients with diabetes and cardiovascular disease",
  "max_docs": 5
}
```

**Response:**
```json
{
  "success": true,
  "query": "elderly patients with diabetes and cardiovascular disease",
  "extracted_info": {
    "relevant_info": [...],
    "summary": "Based on the query...",
    "sources": [...],
    "confidence": 0.85,
    "num_documents": 3
  },
  "filtered_parameters": {
    "relevant_conditions": ["DIABETES", "CARDIOVASCULAR"],
    "relevant_demographics": {"age_range": [65, 100]},
    "relevant_clinical_factors": ["icu", "mortality"]
  }
}
```

### 2. Get RAG Statistics

**GET** `/rag/stats`

Get statistics about the RAG system and vector store.

**Response:**
```json
{
  "status": "initialized",
  "stats": {
    "num_documents": 1250,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 1000,
    "similarity_threshold": 0.6,
    "top_k": 5
  }
}
```

### 3. Add Documents

**POST** `/rag/add-documents`

Add documents to the RAG knowledge base.

**Request Body:**
```json
{
  "texts": [
    "Document 1 content...",
    "Document 2 content..."
  ],
  "metadata": {
    "source": "custom",
    "type": "clinical_guidelines"
  }
}
```

## Integration with Query Parsing

The RAG system is automatically integrated with the query parsing endpoint (`/query/parse`). When you parse a query, the system:

1. Uses RAG to extract relevant information from the knowledge base
2. Filters and enhances the parsed parameters based on retrieved information
3. Includes RAG-extracted information in the response

**Example:**
```bash
POST /query/parse
{
  "query": "Generate data for elderly diabetic patients with hypertension"
}
```

The response will include:
- `parsed_conditions`: Standard parsed filters
- `rag_extracted_info`: Relevant information from the knowledge base
- `research_context`: Enhanced context including RAG information

## Configuration

The RAG system can be configured in `config/settings.py`:

```python
RAG_CONFIG = {
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'similarity_threshold': 0.6,  # Minimum similarity for retrieval
    'top_k': 5,  # Number of documents to retrieve
    'vector_store_path': MODELS_DIR / 'rag_vectorstore'
}
```

## How It Works

1. **Document Ingestion**: Documents are split into chunks and embedded
2. **Vector Store**: Embeddings are stored in ChromaDB for fast similarity search
3. **Query Processing**: User queries are embedded and compared with stored documents
4. **Relevance Filtering**: Only documents above the similarity threshold are returned
5. **Parameter Extraction**: Relevant conditions, demographics, and clinical factors are extracted

## Adding PubMed Articles

PubMed articles are automatically added to the RAG system when:
- You use the `/query/parse` endpoint
- You use the `/generate-context` endpoint

Articles are processed and stored for future queries.

## Adding Custom Documentation

To add your own healthcare documentation:

1. Place markdown or text files in the `backend/docs/` directory
2. The system will automatically load them on startup
3. Or use the `/rag/add-documents` endpoint to add documents programmatically

## Best Practices

1. **Query Specificity**: More specific queries yield better results
   - Good: "elderly diabetic patients with cardiovascular complications"
   - Less effective: "patients"

2. **Document Quality**: Ensure documents are well-formatted and relevant
   - Use clear headings and structure
   - Include relevant medical terminology

3. **Similarity Threshold**: Adjust based on your needs
   - Higher threshold (0.7-0.8): More precise, fewer results
   - Lower threshold (0.5-0.6): More results, may include less relevant info

4. **Chunk Size**: Balance between context and granularity
   - Larger chunks (1500-2000): More context per chunk
   - Smaller chunks (500-800): More granular retrieval

## Troubleshooting

### RAG System Not Initialized

If you see "RAG system not initialized":
1. Check that all dependencies are installed: `pip install -r requirements.txt`
2. Ensure the models directory exists and is writable
3. Check logs for initialization errors

### No Relevant Information Found

If queries return no relevant information:
1. Add more documents to the knowledge base
2. Lower the similarity threshold
3. Check that documents contain relevant content
4. Verify the vector store has documents: `GET /rag/stats`

### Low Confidence Scores

If confidence scores are consistently low:
1. Add more relevant documents
2. Improve document quality and formatting
3. Adjust similarity threshold
4. Use more specific queries

## Example Usage

```python
from src.utils.rag_system import HealthcareRAGSystem

# Initialize RAG system
rag = HealthcareRAGSystem()

# Add documents
rag.add_texts([
    "Diabetes is a chronic condition affecting blood sugar levels...",
    "Cardiovascular disease is the leading cause of death..."
])

# Extract relevant information
result = rag.extract_relevant_info(
    "elderly patients with diabetes and heart disease"
)

print(result['summary'])
print(f"Confidence: {result['confidence']}")
print(f"Found {result['num_documents']} relevant documents")
```

## Performance Considerations

- **Embedding Model**: Uses `all-MiniLM-L6-v2` for fast, efficient embeddings
- **Vector Store**: ChromaDB provides fast similarity search
- **Caching**: Vector store is persisted to disk for faster subsequent queries
- **Batch Processing**: Documents are processed in batches for efficiency

## Future Enhancements

- Support for more embedding models (OpenAI, Cohere, etc.)
- Advanced filtering and ranking
- Multi-query retrieval
- Query expansion and refinement
- Integration with LLMs for answer generation

