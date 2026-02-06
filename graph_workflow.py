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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END

from config import Config
from llm_client import client as _client, openai_retry as _openai_retry
from retriever import SemanticRetriever
from vector_db import VectorDatabase
from confluence_loader import get_confluence_retriever, confluence_available

@_openai_retry
def _llm_call(instructions: str, input_text: str) -> str:
    """LLM call with automatic retry on transient errors."""
    response = _client.responses.create(
        model=Config.OPENAI_MODEL,
        instructions=instructions,
        input=input_text,
        reasoning={"effort": Config.REASONING_EFFORT},
        text={"verbosity": Config.VERBOSITY}
    )
    return response.output_text


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
    
    # Confluence
    include_confluence: bool


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
        
        # Add nodes (analyze + rewrite combined into single node for performance)
        workflow.add_node("analyze_and_rewrite", self._node_analyze_and_rewrite)
        workflow.add_node("retrieve", self._node_retrieve)
        workflow.add_node("grade_documents", self._node_grade_documents)
        workflow.add_node("generate_response", self._node_generate_response)
        
        # Define edges
        workflow.add_edge("analyze_and_rewrite", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # Conditional edge: after grading, either re-query or generate
        workflow.add_conditional_edges(
            "grade_documents",
            self._should_requery,
            {
                "requery": "analyze_and_rewrite",  # Loop back to try again
                "generate": "generate_response"  # Proceed to generation
            }
        )
        
        workflow.add_edge("generate_response", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_and_rewrite")
        
        return workflow.compile()
    
    # ==================== NODE FUNCTIONS ====================
    
    def _node_analyze_and_rewrite(self, state: RAGState) -> Dict[str, Any]:
        """
        Combined node: Analyze query AND generate search variants in ONE LLM call.
        This saves ~5-7 seconds compared to two separate calls.
        """
        _t0 = time.time()
        query = state["original_query"]
        attempt = state.get("retrieval_attempts", 0)
        
        # On re-query attempts, ask for alternative formulations
        if attempt > 0:
            rewrite_instruction = "Generiere ALTERNATIVE Formulierungen, die andere Aspekte betonen."
        else:
            rewrite_instruction = "Generiere Varianten mit Synonymen und unterschiedlichen Perspektiven."
        
        prompt = f"""Analysiere die folgende Anfrage und generiere Such-Varianten.

Anfrage: "{query}"

Aufgaben:
1. Bestimme Query-Typ (factual, comparison, summary, analysis) und Komplexität (simple, moderate, complex)
2. Generiere 3-5 Such-Varianten für besseres Retrieval
   - WICHTIG: Bei MEHREREN Aspekten/Themen erstelle für JEDEN eine separate Query!
   - {rewrite_instruction}

Beispiel für Multi-Aspekt-Anfrage:
- Anfrage: "Vergleiche LOM und IMS LD"
- Queries: ["LOM Eigenschaften", "IMS LD Eigenschaften", "LOM vs IMS LD", "Unterschiede LOM IMS LD"]

Antworte NUR mit diesem JSON-Format:
{{
    "query_type": "factual|comparison|summary|analysis",
    "complexity": "simple|moderate|complex",
    "queries": ["Variante 1", "Variante 2", "Variante 3"]
}}"""
        
        try:
            result_text = _llm_call(
                "Du bist ein Such-Experte. Antworte nur mit validem JSON.",
                prompt
            )
            
            result = json.loads(result_text.strip())
            
            # Extract analysis
            analysis = {
                "query_type": result.get("query_type", "factual"),
                "complexity": result.get("complexity", "moderate"),
                "keywords": query.split()[:5]
            }
            
            # Extract and prepare queries (always include original)
            rewritten = result.get("queries", [])
            rewritten = [query] + [q for q in rewritten[:5] if q != query]
            
        except Exception as e:
            # Fallback: use simple heuristics
            analysis = self._fallback_analysis(query)
            rewritten = [query]
        
        _elapsed = time.time() - _t0
        return {
            "query_analysis": analysis,
            "rewritten_queries": rewritten,
            "workflow_log": [f"[Analyze+Rewrite] {_elapsed:.1f}s - {analysis.get('query_type')}/{analysis.get('complexity')}, {len(rewritten)} variants"]
        }
    
    def _fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Fast heuristic fallback if LLM call fails."""
        query_lower = query.lower()
        
        # Determine query type
        if any(w in query_lower for w in ['vergleich', 'unterschied', 'vs', 'versus']):
            query_type = "comparison"
        elif any(w in query_lower for w in ['zusammenfassung', 'überblick', 'übersicht']):
            query_type = "summary"
        elif any(w in query_lower for w in ['analysiere', 'analyse', 'untersuche']):
            query_type = "analysis"
        else:
            query_type = "factual"
        
        # Determine complexity
        word_count = len(query.split())
        if word_count <= 5:
            complexity = "simple"
        elif ' und ' in query_lower or word_count > 15:
            complexity = "complex"
        else:
            complexity = "moderate"
        
        return {
            "query_type": query_type,
            "complexity": complexity,
            "keywords": query.split()[:5]
        }
    
    def _node_retrieve(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve documents using hybrid search with all query variants."""
        _t0 = time.time()
        queries = state.get("rewritten_queries", [state["original_query"]])
        analysis = state.get("query_analysis", {})
        include_confluence = state.get("include_confluence", False)
        
        # Adaptive k based on complexity - hoch genug für max_chunks (40/80)
        complexity = analysis.get("complexity", "moderate")
        if complexity == "simple":
            top_k = 40
        elif complexity == "complex":
            top_k = 100
        else:
            top_k = 80
        
        # Collect results from all query variants IN PARALLEL
        all_results = {}
        
        def retrieve_for_query(query):
            return self.retriever.hybrid_retrieve_rrf(query, top_k=top_k)
        
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            future_to_query = {executor.submit(retrieve_for_query, q): q for q in queries}
            for future in as_completed(future_to_query):
                results = future.result()
                for doc in results:
                    doc_id = doc.get("id", "")
                    if doc_id not in all_results:
                        all_results[doc_id] = doc
                    else:
                        # Boost score if found by multiple queries
                        existing_score = all_results[doc_id].get("rrf_score", 0)
                        new_score = doc.get("rrf_score", 0)
                        all_results[doc_id]["rrf_score"] = existing_score + new_score * 0.5
        
        # Include Confluence results if enabled
        confluence_count = 0
        confluence_log = []
        if include_confluence:
            confluence_log.append(f"[Confluence] include_confluence=True")
            is_available = confluence_available()
            confluence_log.append(f"[Confluence] confluence_available()={is_available}")
            if is_available:
                confluence_retriever = get_confluence_retriever()
                confluence_log.append(f"[Confluence] Configured: url={confluence_retriever.url}, space={confluence_retriever.space_key}")
            else:
                confluence_log.append("[Confluence] Not available or not configured")
        
        if include_confluence and confluence_available():
            confluence_retriever = get_confluence_retriever()
            
            def search_confluence(query):
                try:
                    return confluence_retriever.search(query, max_results=10)
                except Exception as e:
                    print(f"Confluence search error: {e}")
                    return []
            
            # Parallel Confluence search
            with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                futures = {executor.submit(search_confluence, q): q for q in queries}
                for future in as_completed(futures):
                    confluence_docs = future.result()
                    for doc in confluence_docs:
                        # Validate Confluence result: skip empty or invalid entries
                        doc_text = doc.get("text", "").strip()
                        if not doc_text or len(doc_text) < 20:
                            continue
                        page_id = doc.get('metadata', {}).get('page_id', '')
                        if not page_id:
                            continue
                        # Create unique ID for Confluence docs
                        doc_id = f"confluence_{page_id}"
                        if doc_id not in all_results:
                            all_results[doc_id] = {
                                "id": doc_id,
                                "text": doc_text,
                                "metadata": doc.get("metadata", {}),
                                "rrf_score": 0.5,  # Base score for Confluence docs
                                "source_type": "confluence"
                            }
                            confluence_count += 1
                        else:
                            # Boost if found by multiple queries
                            all_results[doc_id]["rrf_score"] += 0.25
        
        # Sort by combined score and take top results
        sorted_results = sorted(
            all_results.values(), 
            key=lambda x: x.get("rrf_score", 0), 
            reverse=True
        )[:top_k]
        
        _elapsed = time.time() - _t0
        log_msg = f"[Retrieve] {_elapsed:.1f}s - Found {len(sorted_results)} documents using {len(queries)} query variants"
        if confluence_count > 0:
            log_msg += f" (+{confluence_count} from Confluence)"
        
        # Combine all log messages
        all_logs = [log_msg] + confluence_log
        
        return {
            "retrieved_docs": sorted_results,
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
            "workflow_log": all_logs
        }
    
    def _node_grade_documents(self, state: RAGState) -> Dict[str, Any]:
        """
        Rerank documents using Cross-Encoder (no filtering, just reordering).
        Original CRAG filtering was too aggressive - now we keep all docs but reorder them.
        """
        _t0 = time.time()
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
        
        _elapsed = time.time() - _t0
        return {
            "grading_results": grading_results,
            "needs_requery": False,
            "workflow_log": [f"[Rerank] {_elapsed:.1f}s - Reranked {len(reranked_docs)} documents using Cross-Encoder"]
        }
    
    def _node_generate_response(self, state: RAGState) -> Dict[str, Any]:
        """Generate final response using graded documents."""
        _t0 = time.time()
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
            cross_score = doc.get("cross_encoder_score")
            if cross_score is not None:
                # Cross-Encoder scores: -10 bis +10 → normalisieren auf 0-100%
                normalized = (cross_score + 10) / 20  # -10→0, +10→1
                relevance_pct = f"{max(0, min(100, normalized * 100)):.0f}%"
            elif doc.get("similarity_score"):
                relevance_pct = f"{doc['similarity_score']:.0%}"
            else:
                relevance_pct = "N/A"
            
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

Richtlinien:
- Antworte NUR basierend auf den bereitgestellten Dokumenten
- Wenn die Dokumente keine Antwort enthalten, sage das ehrlich
- Zitiere relevante Teile der Dokumente
- Wenn die Frage mehrere Aspekte enthält, gehe auf jeden ein

Format: Schreibe in gut lesbarem Fließtext mit klaren Absätzen. Verwende Aufzählungen nur sparsam, wenn sie wirklich die Lesbarkeit verbessern.

Query-Typ: {query_type}
{length_instruction}"""
        
        user_prompt = f"""DOKUMENTE:
{context}

FRAGE: {query}

Bitte beantworte die Frage basierend auf den obigen Dokumenten in gut lesbarem Fließtext.
{length_instruction}"""
        
        try:
            answer = _llm_call(system_prompt, user_prompt)
        except Exception as e:
            answer = f"Fehler bei der Antwortgenerierung: {str(e)}"
        
        _elapsed = time.time() - _t0
        return {
            "response": answer,
            "sources": sources,
            "workflow_log": [f"[Generate] {_elapsed:.1f}s - Response generated using {len(docs)} documents"]
        }
    
    # ==================== CONDITIONAL EDGES ====================
    
    def _should_requery(self, state: RAGState) -> str:
        """Decide whether to re-query or proceed to generation."""
        if state.get("needs_requery", False):
            return "requery"
        return "generate"
    
    # ==================== PUBLIC INTERFACE ====================
    
    def run(self, query: str, max_attempts: int = 2, response_length: str = "normal", include_confluence: bool = False) -> Dict[str, Any]:
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
            "response_length": response_length,
            "include_confluence": include_confluence
        }
        
        # Execute the graph
        _total_t0 = time.time()
        final_state = self.graph.invoke(initial_state)
        _total_elapsed = time.time() - _total_t0
        
        # Log total time
        print(f"\n{'='*60}")
        print(f"WORKFLOW TIMING SUMMARY (total: {_total_elapsed:.1f}s)")
        for log_entry in final_state.get('workflow_log', []):
            print(f"  {log_entry}")
        print(f"{'='*60}\n")
        
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
