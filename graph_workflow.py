"""
LangGraph-based RAG Workflow

This module implements an agentic RAG pipeline using LangGraph's StateGraph.
The workflow includes:
- Query Analysis
- Query Rewriting (for better retrieval)
- Hybrid Retrieval (RRF: Vector + BM25)
- Document Grading (CRAG)
- Conditional Re-query loop
- Response Generation
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from operator import add
import json

from langgraph.graph import StateGraph, END
from openai import OpenAI

from config import Config
from retriever import SemanticRetriever
from vector_db import VectorDatabase

# Initialize OpenAI client
_client = OpenAI()


class RAGState(TypedDict):
    """State for the RAG workflow graph."""
    # Input
    original_query: str
    
    # Query processing
    rewritten_queries: List[str]
    query_analysis: Dict[str, Any]
    
    # Retrieval
    retrieved_docs: List[Dict[str, Any]]
    grading_results: Dict[str, Any]
    
    # Control flow
    retrieval_attempts: int
    max_retrieval_attempts: int
    needs_requery: bool
    
    # Output
    response: str
    sources: List[Dict[str, Any]]
    
    # Metadata
    workflow_log: Annotated[List[str], add]


class RAGWorkflow:
    """
    LangGraph-based RAG workflow with CRAG (Corrective RAG) pattern.
    
    Workflow:
    1. Analyze Query → Understand intent and complexity
    2. Rewrite Query → Generate alternative phrasings
    3. Retrieve → Hybrid search (Vector + BM25 with RRF)
    4. Grade Documents → Assess relevance (CRAG)
    5. Decision: Re-query or Generate
    6. Generate Response → Create final answer
    """
    
    def __init__(self, retriever: SemanticRetriever = None):
        self.retriever = retriever or SemanticRetriever()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create the graph with our state type
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._node_analyze_query)
        workflow.add_node("rewrite_query", self._node_rewrite_query)
        workflow.add_node("retrieve", self._node_retrieve)
        workflow.add_node("grade_documents", self._node_grade_documents)
        workflow.add_node("generate_response", self._node_generate_response)
        
        # Define edges
        workflow.add_edge("analyze_query", "rewrite_query")
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # Conditional edge: after grading, either re-query or generate
        workflow.add_conditional_edges(
            "grade_documents",
            self._should_requery,
            {
                "requery": "rewrite_query",  # Loop back to try again
                "generate": "generate_response"  # Proceed to generation
            }
        )
        
        workflow.add_edge("generate_response", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        return workflow.compile()
    
    # ==================== NODE FUNCTIONS ====================
    
    def _node_analyze_query(self, state: RAGState) -> Dict[str, Any]:
        """Analyze the query to understand intent and complexity."""
        query = state["original_query"]
        
        prompt = f"""Analysiere die folgende Anfrage und bestimme:
1. Query-Typ (factual, comparison, summary, analysis, recommendation, creative)
2. Schlüssel-Entitäten/Themen
3. Komplexität (simple, moderate, complex)
4. Erwartetes Antwortformat

Anfrage: "{query}"

Antworte NUR mit einem JSON-Objekt:
{{
    "query_type": "...",
    "entities": ["...", "..."],
    "complexity": "...",
    "response_format": "...",
    "keywords": ["...", "..."]
}}"""
        
        try:
            response = _client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Du bist ein Query-Analyse-Experte. Antworte nur mit validem JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            analysis = json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            analysis = {
                "query_type": "unknown",
                "entities": [],
                "complexity": "moderate",
                "response_format": "text",
                "keywords": query.split()[:5]
            }
        
        return {
            "query_analysis": analysis,
            "workflow_log": [f"[Analyze] Query type: {analysis.get('query_type', 'unknown')}, Complexity: {analysis.get('complexity', 'unknown')}"]
        }
    
    def _node_rewrite_query(self, state: RAGState) -> Dict[str, Any]:
        """Rewrite query into multiple variations for better retrieval."""
        query = state["original_query"]
        analysis = state.get("query_analysis", {})
        attempt = state.get("retrieval_attempts", 0)
        
        # On re-query attempts, be more creative with variations
        if attempt > 0:
            instruction = "Generiere ALTERNATIVE Formulierungen, die andere Aspekte der Frage betonen."
        else:
            instruction = "Generiere Varianten mit Synonymen und unterschiedlichen Perspektiven."
        
        prompt = f"""Generiere 3 verschiedene Umformulierungen der folgenden Anfrage für bessere Dokumenten-Suche.
{instruction}

Original-Anfrage: "{query}"
Schlüsselwörter aus Analyse: {analysis.get('keywords', [])}

Antworte NUR mit einem JSON-Array von 3 Strings:
["Variante 1", "Variante 2", "Variante 3"]"""
        
        try:
            response = _client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Du bist ein Query-Rewriting-Experte. Antworte nur mit einem JSON-Array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            rewritten = json.loads(response.choices[0].message.content.strip())
            # Always include original query
            rewritten = [query] + rewritten[:3]
        except Exception as e:
            rewritten = [query]
        
        return {
            "rewritten_queries": rewritten,
            "workflow_log": [f"[Rewrite] Generated {len(rewritten)} query variants"]
        }
    
    def _node_retrieve(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve documents using hybrid search with all query variants."""
        queries = state.get("rewritten_queries", [state["original_query"]])
        analysis = state.get("query_analysis", {})
        
        # Adaptive k based on complexity
        complexity = analysis.get("complexity", "moderate")
        if complexity == "simple":
            top_k = 5
        elif complexity == "complex":
            top_k = 12
        else:
            top_k = 8
        
        # Collect results from all query variants
        all_results = {}
        
        for query in queries:
            results = self.retriever.hybrid_retrieve_rrf(query, top_k=top_k)
            for doc in results:
                doc_id = doc.get("id", "")
                if doc_id not in all_results:
                    all_results[doc_id] = doc
                else:
                    # Boost score if found by multiple queries
                    existing_score = all_results[doc_id].get("rrf_score", 0)
                    new_score = doc.get("rrf_score", 0)
                    all_results[doc_id]["rrf_score"] = existing_score + new_score * 0.5
        
        # Sort by combined score and take top results
        sorted_results = sorted(
            all_results.values(), 
            key=lambda x: x.get("rrf_score", 0), 
            reverse=True
        )[:top_k]
        
        return {
            "retrieved_docs": sorted_results,
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
            "workflow_log": [f"[Retrieve] Found {len(sorted_results)} documents using {len(queries)} query variants"]
        }
    
    def _node_grade_documents(self, state: RAGState) -> Dict[str, Any]:
        """
        Rerank documents using Cross-Encoder (no filtering, just reordering).
        Original CRAG filtering was too aggressive - now we keep all docs but reorder them.
        """
        query = state["original_query"]
        docs = state.get("retrieved_docs", [])
        
        if not docs:
            return {
                "grading_results": {
                    "relevant_docs": [],
                    "ambiguous_docs": [],
                    "irrelevant_count": 0,
                    "needs_web_search": True
                },
                "needs_requery": False,
                "workflow_log": ["[Grade] No documents to grade"]
            }
        
        # Use Cross-Encoder for RERANKING only (not filtering)
        # This keeps all documents but orders them by relevance
        reranked_docs = self.retriever._rerank_documents(query, docs[:8])
        
        # All reranked docs are considered "relevant" (no filtering)
        grading_results = {
            "relevant_docs": reranked_docs,  # Keep ALL docs
            "ambiguous_docs": [],
            "irrelevant_count": 0,
            "needs_web_search": False
        }
        
        return {
            "grading_results": grading_results,
            "needs_requery": False,
            "workflow_log": [f"[Rerank] Reranked {len(reranked_docs)} documents using Cross-Encoder"]
        }
    
    def _node_generate_response(self, state: RAGState) -> Dict[str, Any]:
        """Generate final response using graded documents."""
        query = state["original_query"]
        grading_results = state.get("grading_results", {})
        analysis = state.get("query_analysis", {})
        
        # Combine relevant and ambiguous docs
        docs = grading_results.get("relevant_docs", []) + grading_results.get("ambiguous_docs", [])
        
        if not docs:
            return {
                "response": "Leider konnte ich keine relevanten Informationen zu Ihrer Anfrage finden. Bitte formulieren Sie Ihre Frage um oder laden Sie relevante Dokumente hoch.",
                "sources": [],
                "workflow_log": ["[Generate] No relevant documents found"]
            }
        
        # Prepare context - mehr Chunks für Übersichts-/Zusammenfassungsfragen
        query_type = analysis.get("query_type", "general")
        is_overview_query = query_type in ["summary", "overview", "comparison", "analysis"]
        max_chunks = 80 if is_overview_query else 40
        
        context_parts = []
        sources = []
        
        for i, doc in enumerate(docs[:max_chunks]):
            text = doc.get("text", "")  # Chunks sind bereits auf CHUNK_SIZE begrenzt
            context_parts.append(f"[Dokument {i+1}]:\n{text}")
            
            # Get relevance score from Cross-Encoder or similarity score
            score = doc.get("cross_encoder_score") or doc.get("similarity_score", 0)
            relevance_pct = f"{score:.0%}" if score else "N/A"
            
            sources.append({
                "id": doc.get("id", f"doc_{i}"),
                "text": text[:300] + "...",
                "relevance": relevance_pct
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate response based on query type and length setting
        response_length = state.get("response_length", "normal")
        
        # Length instructions based on setting
        length_instructions = {
            "kurz": "Antworte KURZ und prägnant in 2-3 Sätzen. Nur die wichtigsten Fakten.",
            "normal": "Antworte in angemessener Länge mit den wichtigsten Details.",
            "ausführlich": "Antworte AUSFÜHRLICH und detailliert. Erkläre Zusammenhänge, gib Beispiele und Hintergrundinformationen."
        }
        length_instruction = length_instructions.get(response_length, length_instructions["normal"])
        
        system_prompt = f"""Du bist ein hilfreicher Assistent, der Fragen basierend auf bereitgestellten Dokumenten beantwortet.
- Antworte NUR basierend auf den bereitgestellten Dokumenten
- Wenn die Dokumente keine Antwort enthalten, sage das ehrlich
- Zitiere relevante Teile der Dokumente
- Query-Typ: {query_type}
- WICHTIG: {length_instruction}"""
        
        user_prompt = f"""DOKUMENTE:
{context}

FRAGE: {query}

Bitte beantworte die Frage basierend auf den obigen Dokumenten. {length_instruction}"""
        
        try:
            response = _client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=Config.MAX_TOKENS,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Fehler bei der Antwortgenerierung: {str(e)}"
        
        return {
            "response": answer,
            "sources": sources,
            "workflow_log": [f"[Generate] Response generated using {len(docs)} documents"]
        }
    
    # ==================== CONDITIONAL EDGES ====================
    
    def _should_requery(self, state: RAGState) -> str:
        """Decide whether to re-query or proceed to generation."""
        if state.get("needs_requery", False):
            return "requery"
        return "generate"
    
    # ==================== PUBLIC INTERFACE ====================
    
    def run(self, query: str, max_attempts: int = 2, response_length: str = "normal") -> Dict[str, Any]:
        """
        Run the RAG workflow for a given query.
        
        Args:
            query: User's question
            max_attempts: Maximum retrieval attempts before giving up
            response_length: "kurz", "normal", or "ausführlich"
            
        Returns:
            Dict with response, sources, and workflow metadata
        """
        initial_state = {
            "original_query": query,
            "rewritten_queries": [],
            "query_analysis": {},
            "retrieved_docs": [],
            "grading_results": {},
            "retrieval_attempts": 0,
            "max_retrieval_attempts": max_attempts,
            "needs_requery": False,
            "response": "",
            "sources": [],
            "workflow_log": [],
            "response_length": response_length
        }
        
        # Execute the graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "answer": final_state.get("response", ""),
            "sources": final_state.get("sources", []),
            "query_analysis": final_state.get("query_analysis", {}),
            "grading_summary": {
                "relevant": len(final_state.get("grading_results", {}).get("relevant_docs", [])),
                "ambiguous": len(final_state.get("grading_results", {}).get("ambiguous_docs", [])),
                "irrelevant": final_state.get("grading_results", {}).get("irrelevant_count", 0)
            },
            "retrieval_attempts": final_state.get("retrieval_attempts", 0),
            "workflow_log": final_state.get("workflow_log", [])
        }
    
    def get_graph_visualization(self):
        """Return the graph for visualization (if available)."""
        try:
            return self.graph.get_graph()
        except:
            return None


# Singleton instance for easy import
_workflow_instance = None

def get_rag_workflow(retriever: SemanticRetriever = None) -> RAGWorkflow:
    """Get or create the RAG workflow instance."""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = RAGWorkflow(retriever)
    return _workflow_instance
