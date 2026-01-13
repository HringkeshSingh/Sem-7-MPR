# RAG System Module Structure

This directory contains the modular RAG (Retrieval-Augmented Generation) system implementation.

## 📁 File Structure

```
rag_system/
├── rag_system.py              # Main interface (HealthcareRAGSystem class)
├── rag_core.py                # Core components (embeddings, vectorstore)
├── rag_document_manager.py    # Document management (adding documents)
├── rag_retriever.py           # Information retrieval
└── rag_utils.py               # Helper functions (summary, confidence, filtering)
```

## 🎯 Module Responsibilities

### `rag_system.py` - Main Interface
- **Purpose**: Unified interface for the RAG system
- **Exports**: `HealthcareRAGSystem` class
- **Responsibilities**:
  - Combines all sub-modules
  - Provides public API
  - Maintains backward compatibility

### `rag_core.py` - Core Components
- **Purpose**: Foundation components (embeddings, vector store)
- **Class**: `RAGCore`
- **Responsibilities**:
  - Embedding model initialization
  - Vector store creation/loading
  - Text splitter configuration
  - Retriever setup
  - Vector store statistics
  - Vector store clearing

### `rag_document_manager.py` - Document Management
- **Purpose**: Adding documents to the knowledge base
- **Class**: `RAGDocumentManager`
- **Responsibilities**:
  - Adding raw text documents
  - Adding LangChain Document objects
  - Adding PubMed articles
  - Adding healthcare documentation files
  - Document chunking and processing

### `rag_retriever.py` - Information Retrieval
- **Purpose**: Extracting relevant information from queries
- **Class**: `RAGRetriever`
- **Responsibilities**:
  - Query processing
  - Semantic document retrieval
  - Document processing and formatting
  - Response structure creation

### `rag_utils.py` - Utility Functions
- **Purpose**: Helper functions for RAG operations
- **Functions**:
  - `generate_summary()` - Create human-readable summaries
  - `calculate_confidence()` - Calculate confidence scores
  - `filter_query_parameters()` - Extract relevant parameters

## 🔄 Usage

### Basic Usage (Same as Before)

```python
from src.utils.rag_system import HealthcareRAGSystem

# Initialize (same interface as before)
rag = HealthcareRAGSystem()

# Add documents (same methods)
rag.add_texts(["Document 1", "Document 2"])
rag.add_pubmed_articles(articles)

# Extract information (same methods)
result = rag.extract_relevant_info("diabetes in elderly")
params = rag.filter_query_parameters("query", result)
```

### Advanced Usage (Access Sub-Modules)

```python
# Access core components if needed
embeddings = rag.embeddings
vectorstore = rag.vectorstore
retriever = rag.retriever

# Or import sub-modules directly
from src.utils.rag_core import RAGCore
from src.utils.rag_document_manager import RAGDocumentManager
```

## ✨ Benefits of Modular Structure

1. **Better Readability**: Each file has a single, clear responsibility
2. **Easier Maintenance**: Changes to one component don't affect others
3. **Better Testing**: Each module can be tested independently
4. **Reusability**: Sub-modules can be used independently if needed
5. **Clearer Code Organization**: Related functionality is grouped together

## 📝 Migration Notes

**No changes needed!** The refactoring maintains 100% backward compatibility. All existing code using `HealthcareRAGSystem` will continue to work without modification.

## 🔍 Code Organization Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Separation of Concerns**: Core, documents, retrieval, and utilities are separate
3. **Dependency Injection**: Components are passed to each other, not tightly coupled
4. **Interface Stability**: Public API (`rag_system.py`) remains stable

## 🧪 Testing

Each module can be tested independently:

```python
# Test core components
from src.utils.rag_core import RAGCore
core = RAGCore()
assert core.embeddings is not None

# Test document manager
from src.utils.rag_document_manager import RAGDocumentManager
manager = RAGDocumentManager(core.text_splitter, core.vectorstore)
manager.add_texts(["test"])

# Test retriever
from src.utils.rag_retriever import RAGRetriever
retriever = RAGRetriever(core.retriever)
result = retriever.extract_relevant_info("test query")
```

## 📚 Documentation

- Main RAG guide: `backend/docs/rag_system_guide.md`
- API documentation: See docstrings in each module
- Usage examples: See `rag_system.py` docstrings

