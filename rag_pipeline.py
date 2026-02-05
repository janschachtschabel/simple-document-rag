from typing import Dict, Any, List, Optional, Union
import os
import uuid
from datetime import datetime
from document_processor import DocumentProcessor
from vector_db import VectorDatabase
from retriever import SemanticRetriever
from agent import QueryAgent
from config import Config

class RAGPipeline:
    """
    Main RAG (Retrieval-Augmented Generation) pipeline that orchestrates
    document processing, retrieval, and generation.
    """
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_db = VectorDatabase()
        self.retriever = SemanticRetriever(self.vector_db)
        self.agent = QueryAgent(self.retriever)
        
    def add_document(self, source: Union[str, Dict[str, Any]], 
                    metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Add a document to the RAG system.
        
        Args:
            source: File path, URL, or direct text content
            metadata: Additional metadata for the document
        
        Returns:
            Dictionary containing the result of the operation
        """
        try:
            if isinstance(source, dict) and 'text' in source:
                # Direct text input
                documents = self.document_processor.process_text_direct(
                    source['text'], 
                    metadata or {}
                )
                source_info = "direct_text"
            elif source.startswith(('http://', 'https://')):
                # URL input
                documents = self.document_processor.process_url(source)
                source_info = source
            elif os.path.isfile(source):
                # File input
                documents = self.document_processor.process_file(source)
                source_info = source
            else:
                # Raw text input
                documents = self.document_processor.process_text_direct(source, metadata or {})
                source_info = "raw_text"
            
            # Add additional metadata
            for doc in documents:
                doc['metadata']['added_at'] = datetime.now().isoformat()
                doc['metadata']['document_id'] = str(uuid.uuid4())
                if metadata:
                    doc['metadata'].update(metadata)
            
            # Add to vector database
            doc_ids = self.vector_db.add_documents(documents)
            
            return {
                "success": True,
                "document_ids": doc_ids,
                "source": source_info,
                "chunks_processed": len(documents),
                "message": f"Successfully processed {len(documents)} chunks from {source_info}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "source": source if isinstance(source, str) else "unknown"
            }
    
    def query(self, question: str, 
              retrieval_strategy: str = "semantic",
              top_k: int = None,
              filters: Dict[str, Any] = None,
              include_sources: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: The question to answer
            retrieval_strategy: Strategy for retrieval ("semantic", "hybrid", "rerank")
            top_k: Number of documents to retrieve
            filters: Filters to apply to retrieved documents
            include_sources: Whether to include source information
        
        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            # Retrieve documents based on strategy
            if retrieval_strategy == "hybrid":
                retrieved_docs = self.retriever.hybrid_retrieve(question, top_k)
            elif retrieval_strategy == "rerank":
                retrieved_docs = self.retriever.retrieve_with_reranking(question, top_k)
            else:
                retrieved_docs = self.retriever.retrieve(question, top_k, filters)
            
            # Process query with agent
            agent_result = self.agent.process_query(question, retrieved_docs)
            
            # Prepare response
            response = {
                "question": question,
                "answer": agent_result["response"],
                "retrieval_strategy": retrieval_strategy,
                "documents_retrieved": len(retrieved_docs),
                "query_analysis": agent_result["analysis"],
                "timestamp": datetime.now().isoformat()
            }
            
            if include_sources:
                response["sources"] = agent_result["sources"]
                response["retrieved_documents"] = [
                    {
                        "id": doc["id"],
                        "text": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
                        "metadata": doc["metadata"],
                        "similarity_score": doc.get("similarity_score", 0.0)
                    }
                    for doc in retrieved_docs
                ]
            
            return response
            
        except Exception as e:
            return {
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def add_documents_batch(self, sources: List[Union[str, Dict[str, Any]]], 
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Add multiple documents in batch.
        
        Args:
            sources: List of file paths, URLs, or text content
            metadata: Common metadata for all documents
        
        Returns:
            Dictionary containing batch processing results
        """
        results = []
        total_chunks = 0
        successful = 0
        failed = 0
        
        for source in sources:
            result = self.add_document(source, metadata)
            results.append(result)
            
            if result["success"]:
                successful += 1
                total_chunks += result["chunks_processed"]
            else:
                failed += 1
        
        return {
            "success": True,
            "total_sources": len(sources),
            "successful": successful,
            "failed": failed,
            "total_chunks_processed": total_chunks,
            "results": results
        }
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID.
        """
        return self.vector_db.get_document_by_id(doc_id)
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete a document by ID.
        """
        try:
            success = self.vector_db.delete_document(doc_id)
            return {
                "success": success,
                "document_id": doc_id,
                "message": "Document deleted successfully" if success else "Failed to delete document"
            }
        except Exception as e:
            return {
                "success": False,
                "document_id": doc_id,
                "error": str(e)
            }
    
    def search_documents(self, query: str, 
                       top_k: int = 10,
                       filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Search for documents without generation.
        """
        try:
            results = self.retriever.retrieve(query, top_k, filters)
            
            return {
                "query": query,
                "results": [
                    {
                        "id": doc["id"],
                        "text": doc["text"][:300] + "..." if len(doc["text"]) > 300 else doc["text"],
                        "metadata": doc["metadata"],
                        "similarity_score": doc.get("similarity_score", 0.0)
                    }
                    for doc in results
                ],
                "total_results": len(results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.
        """
        try:
            db_stats = self.vector_db.get_collection_stats()
            
            return {
                "database": db_stats,
                "retrieval_methods": ["semantic", "hybrid", "rerank"],
                "supported_formats": list(Config.SUPPORTED_EXTENSIONS),
                "chunk_size": Config.CHUNK_SIZE,
                "chunk_overlap": Config.CHUNK_OVERLAP,
                "top_k_default": Config.TOP_K_RETRIEVAL,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def clear_all_documents(self) -> Dict[str, Any]:
        """
        Clear all documents from the system.
        """
        try:
            self.vector_db.clear_collection()
            self.agent.clear_history()
            
            return {
                "success": True,
                "message": "All documents and conversation history cleared"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history from the agent.
        """
        return self.agent.get_conversation_history()
    
    def explain_query(self, query: str) -> Dict[str, Any]:
        """
        Get an explanation of how a query would be processed.
        """
        try:
            # Analyze the query
            analysis = self.agent._analyze_query(query)
            
            # Simulate retrieval (without actually retrieving)
            retrieval_info = {
                "strategy": "semantic",
                "expected_results": Config.TOP_K_RETRIEVAL,
                "filters_applied": None
            }
            
            return {
                "query": query,
                "analysis": analysis,
                "retrieval_plan": retrieval_info,
                "processing_steps": [
                    "Query analysis and intent detection",
                    "Document retrieval using semantic search",
                    "Context preparation and augmentation",
                    "Response generation using LLM",
                    "Source citation and formatting"
                ]
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the RAG system.
        """
        try:
            # Test vector database
            db_stats = self.vector_db.get_collection_stats()
            
            # Test embedding engine
            test_embedding = self.retriever.embedding_engine.embed_text("test")
            embedding_dim = len(test_embedding) if hasattr(test_embedding, '__len__') else None
            
            # Test document processor
            test_docs = self.document_processor.process_text_direct("Test document for health check")
            
            return {
                "status": "healthy",
                "components": {
                    "vector_database": "operational",
                    "embedding_engine": "operational",
                    "document_processor": "operational",
                    "retriever": "operational",
                    "agent": "operational"
                },
                "statistics": db_stats,
                "embedding_dimension": embedding_dim,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
