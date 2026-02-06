from openai import OpenAI
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import json
import re
from datetime import datetime
from retriever import SemanticRetriever
from config import Config

# Initialize OpenAI client
_client = OpenAI(api_key=Config.OPENAI_API_KEY)

class QueryType(Enum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    ANALYSIS = "analysis"
    RECOMMENDATION = "recommendation"
    CREATIVE = "creative"
    UNKNOWN = "unknown"

class QueryAgent:
    def __init__(self, retriever: SemanticRetriever = None):
        self.retriever = retriever or SemanticRetriever()
        self.conversation_history = []
        
    def process_query(self, query: str, context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user query using intelligent agent-based approach.
        
        Args:
            query: The user's query
            context: Optional context from previous interactions
        
        Returns:
            Dictionary containing the agent's response and metadata
        """
        # Analyze the query
        query_analysis = self._analyze_query(query)
        
        # Retrieve relevant documents
        retrieved_docs = self._retrieve_documents(query, query_analysis)
        
        # Generate response
        response = self._generate_response(query, query_analysis, retrieved_docs, context)
        
        # Update conversation history
        self._update_history(query, response)
        
        return {
            "query": query,
            "analysis": query_analysis,
            "retrieved_documents": retrieved_docs,
            "response": response,
            "sources": self._extract_sources(retrieved_docs)
        }
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query to understand intent and requirements.
        """
        try:
            prompt = f"""
            Analyze the following query and determine:
            1. Query type (factual, comparison, summary, analysis, recommendation, creative)
            2. Key entities/topics mentioned
            3. Required information type
            4. Complexity level (simple, moderate, complex)
            5. Expected response format
            
            Query: "{query}"
            
            Respond with a JSON object containing these fields:
            {{
                "query_type": "...",
                "entities": ["...", "..."],
                "information_type": "...",
                "complexity": "...",
                "response_format": "...",
                "keywords": ["...", "..."]
            }}
            """
            
            response = _client.responses.create(
                model=Config.OPENAI_MODEL,
                instructions="You are a query analysis expert. Respond only with valid JSON.",
                input=prompt,
                reasoning={"effort": Config.REASONING_EFFORT},
                text={"verbosity": Config.VERBOSITY}
            )
            
            analysis_text = response.output_text.strip()
            
            # Try to parse as JSON
            try:
                analysis = json.loads(analysis_text)
                # Validate and set defaults
                analysis.setdefault('query_type', QueryType.UNKNOWN.value)
                analysis.setdefault('entities', [])
                analysis.setdefault('information_type', 'general')
                analysis.setdefault('complexity', 'moderate')
                analysis.setdefault('response_format', 'paragraph')
                analysis.setdefault('keywords', [])
                
                return analysis
                
            except json.JSONDecodeError:
                # Fallback analysis
                return self._fallback_query_analysis(query)
                
        except Exception as e:
            print(f"Query analysis failed: {str(e)}")
            return self._fallback_query_analysis(query)
    
    def _fallback_query_analysis(self, query: str) -> Dict[str, Any]:
        """
        Fallback query analysis when AI analysis fails.
        """
        query_lower = query.lower()
        
        # Determine query type
        if any(word in query_lower for word in ['what', 'who', 'when', 'where', 'how many']):
            query_type = QueryType.FACTUAL.value
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            query_type = QueryType.COMPARISON.value
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview']):
            query_type = QueryType.SUMMARY.value
        elif any(word in query_lower for word in ['analyze', 'analysis', 'examine']):
            query_type = QueryType.ANALYSIS.value
        elif any(word in query_lower for word in ['recommend', 'should', 'best']):
            query_type = QueryType.RECOMMENDATION.value
        elif any(word in query_lower for word in ['create', 'write', 'generate']):
            query_type = QueryType.CREATIVE.value
        else:
            query_type = QueryType.UNKNOWN.value
        
        # Extract simple keywords
        keywords = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in keywords if len(word) > 3 and word not in ['what', 'how', 'when', 'where', 'why', 'the', 'and', 'for', 'with']]
        
        return {
            "query_type": query_type,
            "entities": keywords[:3],
            "information_type": "general",
            "complexity": "moderate",
            "response_format": "paragraph",
            "keywords": keywords
        }
    
    def _retrieve_documents(self, query: str, analysis: Dict[str, Any], use_grading: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on query analysis.
        Uses RRF hybrid search (Vector + BM25) for better results.
        
        Args:
            query: The search query
            analysis: Query analysis results
            use_grading: If True, use CRAG-style document grading
        """
        query_type = analysis.get('query_type', QueryType.UNKNOWN.value)
        
        if use_grading:
            # CRAG-style retrieval with document grading
            graded_results = self.retriever.retrieve_with_grading(query, top_k=8)
            # Combine relevant and ambiguous docs, prioritizing relevant
            docs = graded_results['relevant_docs'] + graded_results['ambiguous_docs']
            
            # Log grading results
            print(f"CRAG Grading: {len(graded_results['relevant_docs'])} relevant, "
                  f"{len(graded_results['ambiguous_docs'])} ambiguous, "
                  f"{graded_results['irrelevant_count']} irrelevant")
            
            if graded_results['needs_web_search']:
                print("⚠️ Low relevance - consider web search for better results")
            
            return docs[:6]
        
        # Standard retrieval with RRF hybrid search
        if query_type == QueryType.COMPARISON.value:
            # For comparison queries, use RRF + reranking
            docs = self.retriever.hybrid_retrieve_rrf(query, top_k=15)
            docs = self.retriever._rerank_documents(query, docs)[:8]
        elif query_type == QueryType.ANALYSIS.value:
            # For analysis queries, use RRF hybrid search
            docs = self.retriever.hybrid_retrieve_rrf(query, top_k=6)
        elif query_type == QueryType.FACTUAL.value:
            # For factual queries, use RRF for precision
            docs = self.retriever.hybrid_retrieve_rrf(query, top_k=5)
        else:
            # Standard RRF retrieval for other types
            docs = self.retriever.hybrid_retrieve_rrf(query, top_k=5)
        
        return docs
    
    def _generate_response(self, query: str, analysis: Dict[str, Any], 
                          retrieved_docs: List[Dict[str, Any]], 
                          context: List[Dict[str, Any]] = None) -> str:
        """
        Generate a response based on the query and retrieved documents.
        """
        query_type = analysis.get('query_type', QueryType.UNKNOWN.value)
        
        # Prepare context from retrieved documents
        context_text = self._prepare_context(retrieved_docs)
        
        # Prepare conversation context
        conversation_context = self._prepare_conversation_context(context)
        
        # Generate response based on query type
        if query_type == QueryType.FACTUAL.value:
            return self._generate_factual_response(query, context_text, conversation_context)
        elif query_type == QueryType.COMPARISON.value:
            return self._generate_comparison_response(query, context_text, conversation_context)
        elif query_type == QueryType.SUMMARY.value:
            return self._generate_summary_response(query, context_text, conversation_context)
        elif query_type == QueryType.ANALYSIS.value:
            return self._generate_analysis_response(query, context_text, conversation_context)
        elif query_type == QueryType.RECOMMENDATION.value:
            return self._generate_recommendation_response(query, context_text, conversation_context)
        elif query_type == QueryType.CREATIVE.value:
            return self._generate_creative_response(query, context_text, conversation_context)
        else:
            return self._generate_general_response(query, context_text, conversation_context)
    
    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Prepare context text from retrieved documents.
        """
        if not retrieved_docs:
            return "No relevant documents were found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source_info = f"Source {i}: {doc.get('metadata', {}).get('source', 'Unknown')}"
            if 'page' in doc.get('metadata', {}):
                source_info += f" (Page {doc['metadata']['page']})"
            
            context_parts.append(f"{source_info}\n{doc['text']}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _prepare_conversation_context(self, context: List[Dict[str, Any]]) -> str:
        """
        Prepare conversation context from history.
        """
        if not context:
            return ""
        
        context_parts = []
        for item in context[-3:]:  # Use last 3 interactions
            context_parts.append(f"Previous Q: {item.get('query', '')}")
            context_parts.append(f"Previous A: {item.get('response', '')[:200]}...")
        
        return "\n\n".join(context_parts)
    
    def _generate_factual_response(self, query: str, context: str, conversation_context: str) -> str:
        """
        Generate a factual response.
        """
        prompt = f"""
        Based on the provided context, answer the following factual question accurately and concisely.
        If the context doesn't contain the answer, say so clearly.
        
        Question: {query}
        
        Context:
        {context}
        
        {f"Previous conversation:\n{conversation_context}" if conversation_context else ""}
        
        Provide a direct, factual answer with citations to the sources.
        """
        
        return self._call_openai(prompt)
    
    def _generate_comparison_response(self, query: str, context: str, conversation_context: str) -> str:
        """
        Generate a comparison response.
        """
        prompt = f"""
        Based on the provided context, provide a detailed comparison addressing the question.
        Highlight similarities and differences clearly.
        
        Question: {query}
        
        Context:
        {context}
        
        {f"Previous conversation:\n{conversation_context}" if conversation_context else ""}
        
        Structure your response with clear comparison points and cite your sources.
        """
        
        return self._call_openai(prompt)
    
    def _generate_summary_response(self, query: str, context: str, conversation_context: str) -> str:
        """
        Generate a summary response.
        """
        prompt = f"""
        Based on the provided context, provide a comprehensive summary addressing the request.
        Include key points and main ideas.
        
        Request: {query}
        
        Context:
        {context}
        
        {f"Previous conversation:\n{conversation_context}" if conversation_context else ""}
        
        Create a well-structured summary with the most important information.
        """
        
        return self._call_openai(prompt)
    
    def _generate_analysis_response(self, query: str, context: str, conversation_context: str) -> str:
        """
        Generate an analytical response.
        """
        prompt = f"""
        Based on the provided context, provide a detailed analysis addressing the question.
        Include insights, patterns, and deeper understanding.
        
        Question: {query}
        
        Context:
        {context}
        
        {f"Previous conversation:\n{conversation_context}" if conversation_context else ""}
        
        Provide an analytical response with evidence from the sources.
        """
        
        return self._call_openai(prompt)
    
    def _generate_recommendation_response(self, query: str, context: str, conversation_context: str) -> str:
        """
        Generate a recommendation response.
        """
        prompt = f"""
        Based on the provided context, provide recommendations addressing the question.
        Consider pros, cons, and alternatives.
        
        Question: {query}
        
        Context:
        {context}
        
        {f"Previous conversation:\n{conversation_context}" if conversation_context else ""}
        
        Provide well-reasoned recommendations with supporting evidence.
        """
        
        return self._call_openai(prompt)
    
    def _generate_creative_response(self, query: str, context: str, conversation_context: str) -> str:
        """
        Generate a creative response.
        """
        prompt = f"""
        Based on the provided context, create content addressing the request.
        Be creative while staying grounded in the source material.
        
        Request: {query}
        
        Context:
        {context}
        
        {f"Previous conversation:\n{conversation_context}" if conversation_context else ""}
        
        Generate creative content that incorporates information from the sources.
        """
        
        return self._call_openai(prompt)
    
    def _generate_general_response(self, query: str, context: str, conversation_context: str) -> str:
        """
        Generate a general response.
        """
        prompt = f"""
        Based on the provided context, provide a helpful response to the question.
        
        Question: {query}
        
        Context:
        {context}
        
        {f"Previous conversation:\n{conversation_context}" if conversation_context else ""}
        
        Provide a comprehensive and helpful response with citations.
        """
        
        return self._call_openai(prompt)
    
    def _call_openai(self, prompt: str) -> str:
        """
        Call OpenAI API to generate response.
        """
        try:
            response = _client.responses.create(
                model=Config.OPENAI_MODEL,
                instructions="You are a helpful AI assistant that provides accurate, well-sourced responses based on the provided context.",
                input=prompt,
                reasoning={"effort": Config.REASONING_EFFORT},
                text={"verbosity": Config.VERBOSITY}
            )
            
            return response.output_text.strip()
            
        except Exception as e:
            return f"I apologize, but I encountered an error while generating the response: {str(e)}"
    
    def _extract_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source information from retrieved documents.
        """
        sources = []
        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})
            sources.append({
                'id': doc.get('id'),
                'source': metadata.get('source', 'Unknown'),
                'title': metadata.get('title', metadata.get('filename', 'Unknown')),
                'page': metadata.get('page'),
                'similarity_score': doc.get('similarity_score', 0.0)
            })
        
        return sources
    
    def _update_history(self, query: str, response: str):
        """
        Update conversation history.
        """
        self.conversation_history.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 10 interactions
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        """
        return self.conversation_history
    
    def clear_history(self):
        """
        Clear the conversation history.
        """
        self.conversation_history = []
    
    def explain_reasoning(self, query: str, response: str) -> Dict[str, Any]:
        """
        Explain the reasoning behind a response.
        """
        return {
            "query_analysis": self._analyze_query(query),
            "retrieval_strategy": "semantic search with reranking",
            "context_used": len(self.conversation_history),
            "response_type": "context-aware generation"
        }
    
    def generate_document_from_toc(self, toc_entries: List[str], document_title: str = "Generated Document") -> Dict[str, Any]:
        """
        Generate a complete document based on a table of contents.
        Each TOC entry is processed sequentially, with the full TOC context provided to each query.
        
        Args:
            toc_entries: List of TOC entries (chapter/section titles)
            document_title: Title for the generated document
        
        Returns:
            Dictionary containing the generated document and metadata
        """
        if not toc_entries:
            return {"error": "No TOC entries provided", "document": ""}
        
        # Build TOC string for context
        toc_string = "\n".join([f"{i+1}. {entry}" for i, entry in enumerate(toc_entries)])
        
        generated_sections = []
        all_sources = []
        
        for i, entry in enumerate(toc_entries):
            section_result = self._generate_section(
                toc_string=toc_string,
                current_entry=entry,
                entry_index=i + 1,
                total_entries=len(toc_entries),
                previous_sections=generated_sections
            )
            
            generated_sections.append({
                "title": entry,
                "content": section_result["content"],
                "sources": section_result["sources"]
            })
            all_sources.extend(section_result["sources"])
        
        # Combine all sections into final document
        final_document = self._combine_sections(document_title, generated_sections)
        
        return {
            "document_title": document_title,
            "toc": toc_entries,
            "sections": generated_sections,
            "full_document": final_document,
            "total_sources": len(set([s.get('source', '') for s in all_sources])),
            "sources": all_sources
        }
    
    def _generate_section(self, toc_string: str, current_entry: str, entry_index: int, 
                          total_entries: int, previous_sections: List[Dict]) -> Dict[str, Any]:
        """
        Generate content for a single section of the document.
        """
        # Build context from previous sections (summary)
        previous_context = ""
        if previous_sections:
            prev_summaries = []
            for section in previous_sections[-3:]:  # Last 3 sections for context
                prev_summaries.append(f"- {section['title']}: {section['content'][:200]}...")
            previous_context = f"\nPrevious sections covered:\n" + "\n".join(prev_summaries)
        
        # Create a targeted query for RAG retrieval
        retrieval_query = f"{current_entry}"
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve_with_reranking(retrieval_query, top_k=5, rerank_top_n=10)
        
        # Prepare context from retrieved documents
        context_text = self._prepare_context(retrieved_docs)
        
        # Generate section content with full TOC context
        prompt = f"""
Du schreibst einen Abschnitt eines strukturierten Dokuments.

GESAMTES INHALTSVERZEICHNIS:
{toc_string}

AKTUELLER ABSCHNITT ({entry_index} von {total_entries}):
"{current_entry}"
{previous_context}

RELEVANTE INFORMATIONEN AUS DER WISSENSBASIS:
{context_text}

AUFGABE:
Schreibe den Inhalt für den Abschnitt "{current_entry}". 
- Beziehe dich auf die bereitgestellten Informationen aus der Wissensbasis
- Halte den Fokus auf das aktuelle Thema
- Berücksichtige den Kontext des gesamten Dokuments (siehe Inhaltsverzeichnis)
- Schreibe fließenden, gut strukturierten Text
- Wenn keine relevanten Informationen gefunden wurden, gib dies an

Schreibe NUR den Inhalt für diesen Abschnitt, keine Überschrift:
"""
        
        content = self._call_openai(prompt)
        sources = self._extract_sources(retrieved_docs)
        
        return {
            "content": content,
            "sources": sources
        }
    
    def _combine_sections(self, title: str, sections: List[Dict]) -> str:
        """
        Combine all sections into a final formatted document.
        """
        document_parts = [f"# {title}\n"]
        
        # Add TOC
        document_parts.append("## Inhaltsverzeichnis\n")
        for i, section in enumerate(sections, 1):
            document_parts.append(f"{i}. {section['title']}")
        document_parts.append("\n---\n")
        
        # Add sections
        for i, section in enumerate(sections, 1):
            document_parts.append(f"## {i}. {section['title']}\n")
            document_parts.append(section['content'])
            document_parts.append("\n")
        
        return "\n".join(document_parts)
