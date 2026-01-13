# RAG System Troubleshooting Guide

## Error: `'VectorStoreRetriever' object has no attribute 'get_relevant_documents'`

### Root Cause

This error occurs when the LangChain retriever tries to use the old API method `get_relevant_documents()` which doesn't exist in newer LangChain versions (0.1.0+).

### Solution Applied

The code has been updated to:

1. **Always use vectorstore directly** - Bypasses the retriever wrapper entirely
2. **Use `similarity_search_with_score()`** - Direct method call on Chroma vectorstore
3. **Never use retriever if vectorstore is available** - Prevents API compatibility issues

### Verification Steps

1. **Check if vectorstore is initialized:**

   ```python
   # In backend logs, look for:
   "Vectorstore loaded successfully" or "New vectorstore created"
   ```

2. **Check if vectorstore has documents:**

   ```python
   # The code now checks document count and logs it
   # Look for: "Vectorstore has X documents"
   ```

3. **Check backend logs when RAG extraction is called:**
   - Should see: "Attempting vectorstore similarity_search_with_score..."
   - Should see: "✅ Successfully retrieved X documents using similarity_search_with_score"
   - Should NOT see any mention of "get_relevant_documents"

### If Error Persists

1. **Restart the backend server completely:**

   ```bash
   # Stop the server (Ctrl+C)
   # Then restart:
   cd backend
   pipenv run python run_api.py
   ```

2. **Check if vectorstore has documents:**

   - The error might occur if vectorstore is empty
   - Add some documents first using the `/rag/add-documents` endpoint

3. **Check LangChain version:**

   ```bash
   cd backend
   pipenv run pip list | grep langchain
   ```

   Should show:

   - `langchain>=0.1.0`
   - `langchain-community>=0.0.20`
   - `langchain-core>=0.1.0`

4. **Verify vectorstore is being passed:**
   - Check `backend/src/utils/rag_system.py` line ~97
   - Should pass `vectorstore=self._core.vectorstore` to RAGRetriever

### Debug Logging

The updated code now includes extensive logging:

- `logger.debug()` - Shows what method is being tried
- `logger.info()` - Shows success messages
- `logger.error()` - Shows detailed error information
- `logger.exception()` - Shows full traceback

Check backend console/logs for these messages to diagnose the issue.

### Expected Behavior

When RAG extraction is called:

1. ✅ Uses `vectorstore.similarity_search_with_score()` directly
2. ✅ Extracts documents with scores
3. ✅ Calculates confidence based on scores
4. ✅ Returns results without any `get_relevant_documents` errors

### If Still Failing

1. Check backend logs for the exact error message
2. Verify vectorstore initialization in startup logs
3. Ensure vectorstore has documents (check `/rag/stats` endpoint)
4. Try adding a test document to verify vectorstore works
