from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import tempfile
import shutil
from rag_pipeline import RAGPipeline
from config import Config
from graph_workflow import RAGWorkflow
from confluence_loader import get_confluence_retriever, confluence_available

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="A sophisticated Retrieval-Augmented Generation system with intelligent agent-based query processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    retrieval_strategy: str = "semantic"
    top_k: Optional[int] = None
    filters: Optional[Dict[str, Any]] = None
    include_sources: bool = True

class TextDocumentRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class URLDocumentRequest(BaseModel):
    url: str
    metadata: Optional[Dict[str, Any]] = None

class BatchDocumentsRequest(BaseModel):
    sources: List[str]
    metadata: Optional[Dict[str, Any]] = None

class GenerateDocumentRequest(BaseModel):
    toc_entries: List[str]
    document_title: str = "Generated Document"

class GenerateChapterRequest(BaseModel):
    chapter_title: str
    document_title: str = "Document"
    previous_chapters: List[str] = []  # Context from previous chapters

class CRAGQueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    include_ambiguous: bool = True

class LangGraphQueryRequest(BaseModel):
    question: str
    max_attempts: int = 2
    response_length: str = "normal"  # "kurz", "normal", "ausführlich"
    include_confluence: bool = False  # Include Confluence search results

# Routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Agentic RAG API",
        "version": "1.0.0",
        "description": "A sophisticated Retrieval-Augmented Generation system",
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "upload": "/upload",
            "add_text": "/add-text",
            "add_url": "/add-url",
            "query": "/query",
            "search": "/search",
            "documents": "/documents/{doc_id}",
            "conversation": "/conversation",
            "explain": "/explain"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return rag_pipeline.health_check()

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    return rag_pipeline.get_stats()

@app.get("/sources")
async def list_sources():
    """
    List all sources/documents in the database with their chunk counts.
    Returns aggregated info per source file.
    """
    try:
        sources = rag_pipeline.vector_db.list_sources()
        return {
            "sources": sources,
            "total_sources": len(sources),
            "total_chunks": sum(s['chunk_count'] for s in sources)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sources/{source_name:path}")
async def delete_source(source_name: str):
    """
    Delete a source and all its associated chunks from the database.
    
    Args:
        source_name: Name of the source file to delete
    """
    try:
        result = rag_pipeline.vector_db.delete_source(source_name)
        
        # Rebuild BM25 index after deletion
        if result.get('success'):
            try:
                rag_pipeline.retriever.rebuild_bm25_index()
            except Exception as e:
                print(f"BM25 index rebuild warning: {e}")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sources")
async def delete_all_sources():
    """
    Delete ALL sources and chunks from the database.
    Use with caution!
    """
    try:
        rag_pipeline.vector_db.clear_collection()
        
        # Rebuild BM25 index (will be empty)
        try:
            rag_pipeline.retriever.rebuild_bm25_index()
        except Exception as e:
            print(f"BM25 index rebuild warning: {e}")
        
        return {
            "success": True,
            "message": "All sources deleted"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), metadata: str = Form(None)):
    """Upload and process a document file."""
    try:
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in Config.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {Config.SUPPORTED_EXTENSIONS}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Parse metadata if provided
            doc_metadata = {}
            if metadata:
                import json
                doc_metadata = json.loads(metadata)
            
            # Add file metadata
            doc_metadata.update({
                "original_filename": file.filename,
                "file_size": file.size,
                "content_type": file.content_type
            })
            
            # Process the document
            result = rag_pipeline.add_document(temp_file_path, doc_metadata)
            
            # Rebuild BM25 index after adding documents
            try:
                rag_pipeline.retriever.rebuild_bm25_index()
            except Exception as e:
                print(f"BM25 index rebuild warning: {e}")
            
            return result
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-text")
async def add_text_document(request: TextDocumentRequest):
    """Add text content directly."""
    try:
        result = rag_pipeline.add_document(
            {"text": request.text},
            request.metadata or {}
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-url")
async def add_url_document(request: URLDocumentRequest):
    """Add a document from URL."""
    try:
        result = rag_pipeline.add_document(request.url, request.metadata or {})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-batch")
async def add_batch_documents(request: BatchDocumentsRequest):
    """Add multiple documents in batch."""
    try:
        result = rag_pipeline.add_documents_batch(request.sources, request.metadata or {})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_system(request: QueryRequest):
    """Query the RAG system."""
    try:
        result = rag_pipeline.query(
            question=request.question,
            retrieval_strategy=request.retrieval_strategy,
            top_k=request.top_k,
            filters=request.filters,
            include_sources=request.include_sources
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_documents(q: str, top_k: int = 10):
    """Search for documents without generation."""
    try:
        result = rag_pipeline.search_documents(q, top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get a specific document by ID."""
    try:
        document = rag_pipeline.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document by ID."""
    try:
        result = rag_pipeline.delete_document(doc_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation")
async def get_conversation_history():
    """Get conversation history."""
    try:
        history = rag_pipeline.get_conversation_history()
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversation/clear")
async def clear_conversation():
    """Clear conversation history."""
    try:
        rag_pipeline.agent.clear_history()
        return {"message": "Conversation history cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
async def explain_query(question: str = Form(...)):
    """Get an explanation of how a query would be processed."""
    try:
        explanation = rag_pipeline.explain_query(question)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear-all")
async def clear_all_documents():
    """Clear all documents from the system."""
    try:
        result = rag_pipeline.clear_all_documents()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-document")
async def generate_document(request: GenerateDocumentRequest):
    """
    Generate a complete document based on a table of contents.
    Each TOC entry is processed sequentially with RAG retrieval.
    """
    try:
        result = rag_pipeline.agent.generate_document_from_toc(
            toc_entries=request.toc_entries,
            document_title=request.document_title
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-chapter")
async def generate_chapter(request: GenerateChapterRequest):
    """
    Generate a single chapter for incremental document generation.
    Allows progress tracking in the frontend.
    """
    try:
        # Use the workflow for single chapter generation
        workflow = get_langgraph_workflow()
        
        # Create a focused query for this chapter
        query = f"Schreibe einen ausführlichen Abschnitt zum Thema: {request.chapter_title}"
        if request.document_title:
            query = f"Für das Dokument '{request.document_title}': {query}"
        
        result = workflow.run(
            query=query,
            max_attempts=2,
            response_length="ausführlich",
            include_confluence=True
        )
        
        # Format as chapter (workflow returns 'answer' not 'response')
        chapter_content = f"## {request.chapter_title}\n\n{result.get('answer', '')}"
        
        return {
            "chapter_title": request.chapter_title,
            "content": chapter_content,
            "sources": result.get("sources", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-crag")
async def query_with_crag(request: CRAGQueryRequest):
    """
    Query with CRAG (Corrective RAG) - includes document relevance grading.
    Uses RRF hybrid search (Vector + BM25) with LLM-based relevance grading.
    
    Returns documents categorized as relevant, ambiguous, or irrelevant.
    """
    try:
        # Get graded retrieval results
        graded_results = rag_pipeline.retriever.retrieve_with_grading(
            query=request.question,
            top_k=request.top_k
        )
        
        # Prepare documents for response generation
        docs_for_response = graded_results['relevant_docs']
        if request.include_ambiguous:
            docs_for_response += graded_results['ambiguous_docs']
        
        # Generate response if we have relevant docs
        response_text = ""
        if docs_for_response:
            # Analyze query and generate response
            analysis = rag_pipeline.agent._analyze_query(request.question)
            response_text = rag_pipeline.agent._generate_response(
                request.question, analysis, docs_for_response
            )
        else:
            response_text = "Keine relevanten Dokumente gefunden. Bitte formulieren Sie Ihre Anfrage um oder laden Sie relevante Dokumente hoch."
        
        return {
            "answer": response_text,
            "retrieval_info": {
                "method": "CRAG (Corrective RAG) with RRF Hybrid Search",
                "relevant_count": len(graded_results['relevant_docs']),
                "ambiguous_count": len(graded_results['ambiguous_docs']),
                "irrelevant_count": graded_results['irrelevant_count'],
                "needs_web_search": graded_results['needs_web_search']
            },
            "relevant_docs": [{
                "id": doc.get('id'),
                "text": doc.get('text', '')[:500],
                "grade": doc.get('relevance_grade', {})
            } for doc in graded_results['relevant_docs']],
            "ambiguous_docs": [{
                "id": doc.get('id'),
                "text": doc.get('text', '')[:300],
                "grade": doc.get('relevance_grade', {})
            } for doc in graded_results['ambiguous_docs']] if request.include_ambiguous else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-hybrid")
async def search_hybrid(q: str, top_k: int = 10):
    """
    Hybrid search using RRF (Reciprocal Rank Fusion) combining Vector + BM25.
    Better for queries with specific keywords or technical terms.
    """
    try:
        results = rag_pipeline.retriever.hybrid_retrieve_rrf(q, top_k=top_k)
        return {
            "query": q,
            "method": "RRF Hybrid (Vector + BM25)",
            "results": [{
                "id": doc.get('id'),
                "text": doc.get('text', '')[:500],
                "rrf_score": doc.get('rrf_score', 0),
                "metadata": doc.get('metadata', {})
            } for doc in results]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Initialize LangGraph workflow (lazy loading)
_langgraph_workflow = None

def get_langgraph_workflow():
    """Get or create the LangGraph workflow instance."""
    global _langgraph_workflow
    if _langgraph_workflow is None:
        _langgraph_workflow = RAGWorkflow(rag_pipeline.retriever)
    return _langgraph_workflow

@app.post("/query-langgraph")
async def query_with_langgraph(request: LangGraphQueryRequest):
    """
    Query using LangGraph-based workflow with:
    - Query Analysis
    - Query Rewriting (multiple variations)
    - Hybrid Retrieval (RRF: Vector + BM25)
    - Document Grading (CRAG)
    - Conditional Re-query loop
    - Response Generation
    
    This is the most advanced retrieval mode with automatic retry on low-quality results.
    """
    try:
        workflow = get_langgraph_workflow()
        result = workflow.run(
            query=request.question,
            max_attempts=request.max_attempts,
            response_length=request.response_length,
            include_confluence=request.include_confluence
        )
        
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "metadata": {
                "query_analysis": result.get("query_analysis", {}),
                "grading_summary": result.get("grading_summary", {}),
                "retrieval_attempts": result.get("retrieval_attempts", 0),
                "workflow_log": result.get("workflow_log", [])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ConfluenceConfigRequest(BaseModel):
    url: str
    username: str
    api_key: str
    space_key: str = ""

@app.get("/confluence/status")
async def confluence_status():
    """Check Confluence integration status."""
    retriever = get_confluence_retriever()
    return {
        "available": confluence_available(),
        "configured": retriever.is_configured(),
        "url": retriever.url if retriever.url else None,
        "space_key": retriever.space_key if retriever.space_key else None
    }

@app.post("/confluence/config")
async def save_confluence_config(request: ConfluenceConfigRequest):
    """Save Confluence configuration (runtime only, not persisted to .env)."""
    try:
        retriever = get_confluence_retriever()
        retriever.url = request.url
        retriever.username = request.username
        retriever.api_key = request.api_key
        retriever.space_key = request.space_key
        retriever._loader = None  # Reset loader to use new config
        retriever._initialized = False
        
        return {
            "success": True,
            "message": "Confluence configuration saved",
            "configured": retriever.is_configured()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/confluence/search")
async def confluence_search(q: str, max_results: int = 5):
    """Search Confluence for documents."""
    if not confluence_available():
        raise HTTPException(status_code=503, detail="Confluence not configured")
    
    retriever = get_confluence_retriever()
    results = retriever.search(q, max_results=max_results)
    
    return {
        "query": q,
        "results": results,
        "count": len(results)
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )

if __name__ == "__main__":
    import uvicorn
    Config.validate()
    uvicorn.run(app, host="0.0.0.0", port=8000)
