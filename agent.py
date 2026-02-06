from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import json
import re
from datetime import datetime
from retriever import SemanticRetriever
from config import Config
from llm_client import client as _client, openai_retry as _openai_retry

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
            Analysiere die folgende Anfrage und bestimme:
            1. Anfrage-Typ (factual, comparison, summary, analysis, recommendation, creative)
            2. Wichtige Entitäten/Themen
            3. Benötigter Informationstyp
            4. Komplexitätsgrad (simple, moderate, complex)
            5. Erwartetes Antwortformat
            
            Anfrage: "{query}"
            
            Antworte mit einem JSON-Objekt:
            {{
                "query_type": "...",
                "entities": ["...", "..."],
                "information_type": "...",
                "complexity": "...",
                "response_format": "...",
                "keywords": ["...", "..."]
            }}
            """
            
            response = _openai_retry(
                lambda: _client.responses.create(
                    model=Config.OPENAI_MODEL,
                    instructions="Du bist ein Experte für Query-Analyse. Antworte nur mit validem JSON.",
                    input=prompt,
                    reasoning={"effort": Config.REASONING_EFFORT},
                    text={"verbosity": Config.VERBOSITY}
                )
            )()
            
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
        if any(word in query_lower for word in ['was', 'wer', 'wann', 'wo', 'wie viele', 'welche', 'what', 'who', 'when', 'where']):
            query_type = QueryType.FACTUAL.value
        elif any(word in query_lower for word in ['vergleich', 'unterschied', 'versus', 'vs', 'compare', 'difference']):
            query_type = QueryType.COMPARISON.value
        elif any(word in query_lower for word in ['zusammenfassung', 'überblick', 'übersicht', 'summarize', 'summary', 'overview']):
            query_type = QueryType.SUMMARY.value
        elif any(word in query_lower for word in ['analysiere', 'analyse', 'untersuche', 'analyze', 'analysis']):
            query_type = QueryType.ANALYSIS.value
        elif any(word in query_lower for word in ['empfehlung', 'empfiehl', 'sollte', 'beste', 'recommend', 'should', 'best']):
            query_type = QueryType.RECOMMENDATION.value
        elif any(word in query_lower for word in ['erstelle', 'schreibe', 'generiere', 'create', 'write', 'generate']):
            query_type = QueryType.CREATIVE.value
        else:
            query_type = QueryType.UNKNOWN.value
        
        # Extract simple keywords (German + English stopwords)
        keywords = re.findall(r'\b\w+\b', query.lower())
        stopwords = ['was', 'wie', 'wann', 'wo', 'warum', 'der', 'die', 'das', 'und', 'für', 'mit', 'ein', 'eine', 'ist', 'sind', 'what', 'how', 'when', 'where', 'why', 'the', 'and', 'for', 'with']
        keywords = [word for word in keywords if len(word) > 3 and word not in stopwords]
        
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
        Beantworte die folgende Frage basierend auf dem bereitgestellten Kontext genau und präzise.
        Wenn der Kontext die Antwort nicht enthält, sage das klar.
        
        Frage: {query}
        
        Kontext:
        {context}
        
        {f"Vorheriges Gespräch:\n{conversation_context}" if conversation_context else ""}
        
        Gib eine direkte, faktische Antwort mit Verweisen auf die Quellen.
        """
        
        return self._call_openai(prompt)
    
    def _generate_comparison_response(self, query: str, context: str, conversation_context: str) -> str:
        """
        Generate a comparison response.
        """
        prompt = f"""
        Erstelle basierend auf dem bereitgestellten Kontext einen detaillierten Vergleich zur Frage.
        Hebe Gemeinsamkeiten und Unterschiede klar hervor.
        
        Frage: {query}
        
        Kontext:
        {context}
        
        {f"Vorheriges Gespräch:\n{conversation_context}" if conversation_context else ""}
        
        Strukturiere deine Antwort mit klaren Vergleichspunkten und zitiere die Quellen.
        """
        
        return self._call_openai(prompt)
    
    def _generate_summary_response(self, query: str, context: str, conversation_context: str) -> str:
        """
        Generate a summary response.
        """
        prompt = f"""
        Erstelle basierend auf dem bereitgestellten Kontext eine umfassende Zusammenfassung.
        Nenne die wichtigsten Punkte und Kernaussagen.
        
        Anfrage: {query}
        
        Kontext:
        {context}
        
        {f"Vorheriges Gespräch:\n{conversation_context}" if conversation_context else ""}
        
        Erstelle eine gut strukturierte Zusammenfassung mit den wichtigsten Informationen.
        """
        
        return self._call_openai(prompt)
    
    def _generate_analysis_response(self, query: str, context: str, conversation_context: str) -> str:
        """
        Generate an analytical response.
        """
        prompt = f"""
        Erstelle basierend auf dem bereitgestellten Kontext eine detaillierte Analyse zur Frage.
        Berücksichtige Erkenntnisse, Muster und tiefergehende Zusammenhänge.
        
        Frage: {query}
        
        Kontext:
        {context}
        
        {f"Vorheriges Gespräch:\n{conversation_context}" if conversation_context else ""}
        
        Liefere eine analytische Antwort mit Belegen aus den Quellen.
        """
        
        return self._call_openai(prompt)
    
    def _generate_recommendation_response(self, query: str, context: str, conversation_context: str) -> str:
        """
        Generate a recommendation response.
        """
        prompt = f"""
        Gib basierend auf dem bereitgestellten Kontext Empfehlungen zur Frage.
        Berücksichtige Vor- und Nachteile sowie Alternativen.
        
        Frage: {query}
        
        Kontext:
        {context}
        
        {f"Vorheriges Gespräch:\n{conversation_context}" if conversation_context else ""}
        
        Liefere gut begründete Empfehlungen mit Belegen.
        """
        
        return self._call_openai(prompt)
    
    def _generate_creative_response(self, query: str, context: str, conversation_context: str) -> str:
        """
        Generate a creative response.
        """
        prompt = f"""
        Erstelle basierend auf dem bereitgestellten Kontext kreativen Inhalt zur Anfrage.
        Sei kreativ, bleibe aber bei den Quellen.
        
        Anfrage: {query}
        
        Kontext:
        {context}
        
        {f"Vorheriges Gespräch:\n{conversation_context}" if conversation_context else ""}
        
        Generiere kreativen Inhalt, der Informationen aus den Quellen einbezieht.
        """
        
        return self._call_openai(prompt)
    
    def _generate_general_response(self, query: str, context: str, conversation_context: str) -> str:
        """
        Generate a general response.
        """
        prompt = f"""
        Beantworte basierend auf dem bereitgestellten Kontext die folgende Frage hilfreich.
        
        Frage: {query}
        
        Kontext:
        {context}
        
        {f"Vorheriges Gespräch:\n{conversation_context}" if conversation_context else ""}
        
        Gib eine umfassende und hilfreiche Antwort mit Quellenverweisen.
        """
        
        return self._call_openai(prompt)
    
    def _call_openai(self, prompt: str) -> str:
        """
        Call OpenAI API to generate response.
        """
        try:
            response = _openai_retry(
                lambda: _client.responses.create(
                    model=Config.OPENAI_MODEL,
                    instructions="Du bist ein hilfreicher KI-Assistent, der genaue, quellenbasierte Antworten auf Basis des bereitgestellten Kontexts gibt.",
                    input=prompt,
                    reasoning={"effort": Config.REASONING_EFFORT},
                    text={"verbosity": Config.VERBOSITY}
                )
            )()
            
            return response.output_text.strip()
            
        except Exception as e:
            return f"Es ist ein Fehler bei der Antwortgenerierung aufgetreten: {str(e)}"
    
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
