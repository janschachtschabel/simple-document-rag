from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from rank_bm25 import BM25Okapi
from vector_db import VectorDatabase
from embeddings import EmbeddingEngine
from config import Config
from llm_client import client as _client, openai_retry as _openai_retry

# Cross-Encoder for fast reranking (German support)
try:
    from sentence_transformers import CrossEncoder
    # Multilingual model with German support - trained on mMARCO
    CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    _cross_encoder = None
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    _cross_encoder = None
    print("Warning: CrossEncoder not available. Install sentence-transformers.")

def get_cross_encoder():
    """Lazy load Cross-Encoder model."""
    global _cross_encoder
    if CROSS_ENCODER_AVAILABLE and _cross_encoder is None:
        print(f"Loading Cross-Encoder: {CROSS_ENCODER_MODEL}")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
        print("Cross-Encoder loaded successfully")
    return _cross_encoder

class SemanticRetriever:
    def __init__(self, vector_db: VectorDatabase = None):
        self.vector_db = vector_db or VectorDatabase()
        self.embedding_engine = EmbeddingEngine()
        self._bm25_index = None
        self._bm25_docs = None
        self._bm25_doc_ids = None
        # Build BM25 index once at startup
        self._build_bm25_index()
    
    def expand_with_neighbors(self, documents: List[Dict[str, Any]], window: int = 1) -> List[Dict[str, Any]]:
        """
        Expand retrieved documents with neighboring chunks for more context.
        
        Args:
            documents: Retrieved documents
            window: Number of chunks before/after to include (default: 1)
        
        Returns:
            Documents with expanded context from neighboring chunks
        """
        if not documents:
            return documents
        
        expanded_docs = []
        seen_ids = set()
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            source = metadata.get('source', '')
            chunk_index = metadata.get('chunk_index')
            total_chunks = metadata.get('total_chunks', 0)
            
            # Skip if no chunk info available
            if chunk_index is None or not source:
                expanded_docs.append(doc)
                seen_ids.add(doc.get('id', ''))
                continue
            
            # Calculate neighbor indices
            neighbor_indices = []
            for offset in range(-window, window + 1):
                idx = chunk_index + offset
                if 0 <= idx < total_chunks:
                    neighbor_indices.append(idx)
            
            # Fetch neighbors from same source
            neighbor_texts = []
            for idx in neighbor_indices:
                if idx == chunk_index:
                    # Current chunk
                    neighbor_texts.append(doc.get('text', ''))
                else:
                    # Try to fetch neighbor chunk
                    neighbor = self._fetch_chunk_by_index(source, idx)
                    if neighbor and neighbor.get('id') not in seen_ids:
                        neighbor_texts.append(neighbor.get('text', ''))
                        seen_ids.add(neighbor.get('id', ''))
            
            # Create expanded document
            expanded_doc = doc.copy()
            if len(neighbor_texts) > 1:
                expanded_doc['text'] = '\n\n[...]\n\n'.join(neighbor_texts)
                expanded_doc['metadata'] = metadata.copy()
                expanded_doc['metadata']['expanded'] = True
                expanded_doc['metadata']['chunks_included'] = len(neighbor_texts)
            
            if doc.get('id') not in seen_ids:
                expanded_docs.append(expanded_doc)
                seen_ids.add(doc.get('id', ''))
        
        return expanded_docs
    
    def _fetch_chunk_by_index(self, source: str, chunk_index: int) -> Optional[Dict[str, Any]]:
        """Fetch a specific chunk by source and index."""
        try:
            results = self.vector_db.collection.get(
                where={
                    "$and": [
                        {"source": {"$eq": source}},
                        {"chunk_index": {"$eq": chunk_index}}
                    ]
                },
                include=['documents', 'metadatas']
            )
            
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'text': results['documents'][0] if results['documents'] else '',
                    'metadata': results['metadatas'][0] if results['metadatas'] else {}
                }
        except Exception as e:
            pass  # Silently fail for neighbor fetching
        
        return None
    
    def retrieve(self, query: str, top_k: int = None, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The query string
            top_k: Number of documents to retrieve
            filters: Optional filters to apply to results
        
        Returns:
            List of retrieved documents with scores
        """
        top_k = top_k or Config.TOP_K_RETRIEVAL
        
        # Perform semantic search
        results = self.vector_db.query(query, n_results=top_k)
        
        # Apply filters if provided
        if filters:
            filtered_results = []
            for result in results['results']:
                if self._matches_filters(result['metadata'], filters):
                    filtered_results.append(result)
            results['results'] = filtered_results
        
        # Add relevance scores
        for result in results['results']:
            if result.get('distance') is not None:
                # Convert distance to similarity score (lower distance = higher similarity)
                result['similarity_score'] = 1 - result['distance']
            else:
                result['similarity_score'] = 0.0
        
        return results['results']
    
    def retrieve_with_reranking(self, query: str, top_k: int = None, rerank_top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve documents with reranking for better relevance.
        
        Args:
            query: The query string
            top_k: Final number of documents to return
            rerank_top_n: Number of documents to consider for reranking
        
        Returns:
            List of reranked documents
        """
        top_k = top_k or Config.TOP_K_RETRIEVAL
        
        # First, retrieve more documents
        initial_results = self.retrieve(query, top_k=rerank_top_n)
        
        if len(initial_results) <= top_k:
            return initial_results
        
        # Rerank using cross-encoder or other method
        reranked_results = self._rerank_documents(query, initial_results)
        
        return reranked_results[:top_k]
    
    def hybrid_retrieve(self, query: str, top_k: int = None, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining semantic and keyword search.
        
        Args:
            query: The query string
            top_k: Number of documents to retrieve
            semantic_weight: Weight for semantic search (0-1)
        
        Returns:
            List of hybrid retrieved documents
        """
        top_k = top_k or Config.TOP_K_RETRIEVAL
        
        # Semantic search
        semantic_results = self.retrieve(query, top_k=top_k * 2)
        
        # Keyword search (simple implementation)
        keyword_results = self._keyword_search(query, top_k=top_k * 2)
        
        # Combine and re-rank
        combined_results = self._combine_results(
            semantic_results, 
            keyword_results, 
            semantic_weight
        )
        
        return combined_results[:top_k]
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document metadata matches the provided filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, (list, tuple)):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    def _rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank documents using Cross-Encoder (fast, ~200ms for 10 docs).
        Falls back to LLM-based reranking if Cross-Encoder unavailable.
        """
        if len(documents) <= 1:
            return documents
        
        # Try Cross-Encoder first (fast)
        cross_encoder = get_cross_encoder()
        if cross_encoder is not None:
            return self._rerank_with_cross_encoder(query, documents, cross_encoder)
        
        # Fallback to LLM-based reranking (slow)
        return self._rerank_with_llm(query, documents)
    
    def _rerank_with_cross_encoder(self, query: str, documents: List[Dict[str, Any]], 
                                    cross_encoder) -> List[Dict[str, Any]]:
        """
        Fast reranking using Cross-Encoder model.
        Processes all documents in a single batch (~200ms).
        """
        try:
            # Prepare query-document pairs
            pairs = [(query, doc.get('text', '')[:512]) for doc in documents]
            
            # Get scores in batch (fast!)
            scores = cross_encoder.predict(pairs)
            
            # Add scores to documents
            for i, doc in enumerate(documents):
                doc['cross_encoder_score'] = float(scores[i])
            
            # Sort by Cross-Encoder score (descending)
            reranked = sorted(documents, key=lambda x: x.get('cross_encoder_score', 0), reverse=True)
            
            return reranked
            
        except Exception as e:
            print(f"Cross-Encoder reranking failed: {str(e)}")
            return documents
    
    def _rerank_with_llm(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback LLM-based reranking (slower, ~2-4s).
        """
        try:
            doc_texts = [doc['text'] for doc in documents]
            
            prompt = f"""
            Please rank the following documents by their relevance to the query: "{query}"
            
            Documents:
            {chr(10).join([f"{i+1}. {text[:200]}..." for i, text in enumerate(doc_texts)])}
            
            Return only the ranking numbers (e.g., "3,1,4,2") without any explanation.
            """
            
            response = _openai_retry(
                lambda: _client.responses.create(
                    model=Config.OPENAI_MODEL,
                    instructions="Du bist ein hilfreicher Assistent, der Dokumente nach Relevanz sortiert.",
                    input=prompt,
                    reasoning={"effort": Config.REASONING_EFFORT},
                    text={"verbosity": Config.VERBOSITY}
                )
            )()
            
            ranking_text = response.output_text.strip()
            rankings = [int(x.strip()) - 1 for x in ranking_text.split(',')]
            
            if len(rankings) == len(documents) and set(rankings) == set(range(len(documents))):
                reranked = [documents[i] for i in rankings]
                return reranked
            
        except Exception as e:
            print(f"LLM reranking failed: {str(e)}")
        
        return documents
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from all documents in vector database."""
        try:
            # Get all documents from ChromaDB
            all_docs = self.vector_db.collection.get(include=['documents', 'metadatas'])
            
            if not all_docs['documents']:
                print("No documents found for BM25 index")
                return
            
            self._bm25_docs = []
            self._bm25_doc_ids = all_docs['ids']
            tokenized_docs = []
            
            for i, doc_text in enumerate(all_docs['documents']):
                self._bm25_docs.append({
                    'id': all_docs['ids'][i],
                    'text': doc_text,
                    'metadata': all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                })
                tokenized_docs.append(self._tokenize(doc_text))
            
            self._bm25_index = BM25Okapi(tokenized_docs)
            print(f"BM25 index built with {len(tokenized_docs)} documents")
            
        except Exception as e:
            print(f"Failed to build BM25 index: {str(e)}")
            self._bm25_index = None
    
    def _keyword_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        BM25-based keyword search implementation.
        Uses Okapi BM25 algorithm for better ranking than simple keyword overlap.
        """
        try:
            # Build BM25 index if not exists
            if self._bm25_index is None:
                self._build_bm25_index()
            
            if self._bm25_index is None or not self._bm25_docs:
                return []
            
            # Tokenize query and get BM25 scores
            query_tokens = self._tokenize(query)
            scores = self._bm25_index.get_scores(query_tokens)
            
            # Create results with scores
            keyword_results = []
            for i, score in enumerate(scores):
                if score > 0:  # Only include docs with positive BM25 score
                    result = self._bm25_docs[i].copy()
                    result['keyword_score'] = float(score)
                    result['bm25_score'] = float(score)
                    keyword_results.append(result)
            
            # Sort by BM25 score
            keyword_results.sort(key=lambda x: x['keyword_score'], reverse=True)
            
            return keyword_results[:top_k]
            
        except Exception as e:
            print(f"BM25 keyword search failed: {str(e)}")
            return []
    
    def _combine_results(self, semantic_results: List[Dict[str, Any]], 
                        keyword_results: List[Dict[str, Any]], 
                        semantic_weight: float) -> List[Dict[str, Any]]:
        """
        Combine semantic and keyword search results.
        """
        # Create a mapping of document ID to results
        combined = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = result['id']
            combined[doc_id] = result.copy()
            combined[doc_id]['semantic_score'] = result.get('similarity_score', 0.0)
            combined[doc_id]['keyword_score'] = 0.0
        
        # Add keyword results
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in combined:
                combined[doc_id]['keyword_score'] = result.get('keyword_score', 0.0)
            else:
                combined[doc_id] = result.copy()
                combined[doc_id]['semantic_score'] = 0.0
                combined[doc_id]['keyword_score'] = result.get('keyword_score', 0.0)
        
        # Calculate combined scores
        for doc_id, result in combined.items():
            combined_score = (
                semantic_weight * result['semantic_score'] + 
                (1 - semantic_weight) * result['keyword_score']
            )
            result['combined_score'] = combined_score
        
        # Sort by combined score
        final_results = list(combined.values())
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return final_results
    
    def get_similar_documents(self, doc_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given document.
        """
        try:
            # Get the document
            doc = self.vector_db.get_document_by_id(doc_id)
            if not doc:
                return []
            
            # Use the document text as a query
            similar_docs = self.retrieve(doc['text'], top_k=top_k)
            
            # Remove the original document from results
            similar_docs = [doc for doc in similar_docs if doc['id'] != doc_id]
            
            return similar_docs
            
        except Exception as e:
            print(f"Error finding similar documents: {str(e)}")
            return []
    
    def explain_retrieval(self, query: str, doc_id: str) -> Dict[str, Any]:
        """
        Provide an explanation for why a document was retrieved for a query.
        """
        try:
            # Get the document
            doc = self.vector_db.get_document_by_id(doc_id)
            if not doc:
                return {"error": "Document not found"}
            
            # Calculate similarity
            query_embedding = self.embedding_engine.embed_text(query)
            doc_embedding = self.embedding_engine.embed_text(doc['text'])
            
            similarity = self.embedding_engine.compute_similarity(
                query_embedding, [doc_embedding]
            )[0]
            
            # Find common terms
            query_terms = set(query.lower().split())
            doc_terms = set(doc['text'].lower().split())
            common_terms = query_terms.intersection(doc_terms)
            
            return {
                "query": query,
                "document_id": doc_id,
                "similarity_score": similarity,
                "common_terms": list(common_terms),
                "explanation": f"Document retrieved with similarity score {similarity:.3f}. " +
                              f"Common terms: {', '.join(common_terms[:5])}"
            }
            
        except Exception as e:
            return {"error": f"Explanation failed: {str(e)}"}
    
    def rebuild_bm25_index(self) -> None:
        """Force rebuild of BM25 index after documents are added."""
        self._bm25_index = None
        self._bm25_docs = None
        self._bm25_doc_ids = None
        self._build_bm25_index()
    
    def hybrid_retrieve_rrf(self, query: str, top_k: int = None, k: int = 60) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval using Reciprocal Rank Fusion (RRF).
        RRF combines rankings from multiple retrieval methods more robustly than weighted averaging.
        
        Formula: RRF_score = sum(1 / (k + rank_i)) for each retrieval method i
        
        Args:
            query: The query string
            top_k: Number of documents to retrieve
            k: RRF parameter (default 60, standard value)
        
        Returns:
            List of hybrid retrieved documents with RRF scores
        """
        top_k = top_k or Config.TOP_K_RETRIEVAL
        
        # Get results from both methods IN PARALLEL
        _t0 = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            sem_future = executor.submit(self.retrieve, query, top_k * 3)
            bm25_future = executor.submit(self._keyword_search, query, top_k * 3)
            semantic_results = sem_future.result()
            keyword_results = bm25_future.result()
        _t_total = time.time() - _t0
        
        print(f"  [RRF] parallel={_t_total:.2f}s, top_k={top_k}, sem={len(semantic_results)}, bm25={len(keyword_results)}")
        
        # Calculate RRF scores
        rrf_scores = {}
        
        # Add semantic results with rank
        for rank, result in enumerate(semantic_results):
            doc_id = result['id']
            rrf_scores[doc_id] = {
                'doc': result,
                'rrf_score': 1.0 / (k + rank + 1)  # +1 because rank is 0-indexed
            }
        
        # Add keyword results with rank
        for rank, result in enumerate(keyword_results):
            doc_id = result['id']
            rrf_contribution = 1.0 / (k + rank + 1)
            
            if doc_id in rrf_scores:
                rrf_scores[doc_id]['rrf_score'] += rrf_contribution
            else:
                rrf_scores[doc_id] = {
                    'doc': result,
                    'rrf_score': rrf_contribution
                }
        
        # Sort by RRF score and return
        sorted_results = sorted(rrf_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
        
        final_results = []
        for item in sorted_results[:top_k]:
            doc = item['doc'].copy()
            doc['rrf_score'] = item['rrf_score']
            final_results.append(doc)
        
        return final_results
    
    def grade_document_relevance(self, query: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Grade document relevance using Cross-Encoder score thresholds.
        Fast alternative to LLM-based CRAG grading (~20ms vs ~500ms per doc).
        
        Returns: 'relevant', 'ambiguous', or 'irrelevant' based on score thresholds.
        """
        cross_encoder = get_cross_encoder()
        
        if cross_encoder is not None:
            return self._grade_with_cross_encoder(query, document, cross_encoder)
        
        # Fallback to LLM-based grading (slow)
        return self._grade_with_llm(query, document)
    
    def _grade_with_cross_encoder(self, query: str, document: Dict[str, Any], 
                                   cross_encoder) -> Dict[str, Any]:
        """
        Fast grading using Cross-Encoder score thresholds.
        Thresholds calibrated for mmarco-mMiniLMv2 model.
        """
        try:
            doc_text = document.get('text', '')[:512]
            score = float(cross_encoder.predict([(query, doc_text)])[0])
            
            # Thresholds for mmarco model (scores typically -10 to +10)
            if score > 2.0:
                grade = "relevant"
                confidence = min(1.0, (score - 2.0) / 8.0 + 0.7)
            elif score > -1.0:
                grade = "ambiguous"
                confidence = 0.5 + (score + 1.0) / 6.0
            else:
                grade = "irrelevant"
                confidence = min(1.0, abs(score) / 10.0)
            
            return {
                "grade": grade,
                "confidence": round(confidence, 3),
                "score": round(score, 3),
                "method": "cross_encoder",
                "document_id": document.get('id', 'unknown')
            }
            
        except Exception as e:
            print(f"Cross-Encoder grading failed: {str(e)}")
            return {"grade": "ambiguous", "confidence": 0.5, "document_id": document.get('id', 'unknown')}
    
    def _grade_with_llm(self, query: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback LLM-based grading (slower but more nuanced).
        """
        try:
            doc_text = document.get('text', '')[:1500]
            
            prompt = f"""Bewerte die Relevanz des folgenden Dokuments für die gegebene Anfrage.

ANFRAGE: "{query}"

DOKUMENT:
{doc_text}

Bewerte mit einer der folgenden Kategorien:
- "relevant": Das Dokument enthält direkt nützliche Informationen zur Beantwortung der Anfrage
- "ambiguous": Das Dokument könnte relevant sein, aber die Verbindung ist unklar oder indirekt
- "irrelevant": Das Dokument hat keine Verbindung zur Anfrage

Antworte NUR mit einem JSON-Objekt:
{{"grade": "relevant|ambiguous|irrelevant", "confidence": 0.0-1.0, "reason": "kurze Begründung"}}"""

            response = _openai_retry(
                lambda: _client.responses.create(
                    model=Config.OPENAI_MODEL,
                    instructions="Du bist ein Experte für Dokumentenrelevanz-Bewertung. Antworte nur mit validem JSON.",
                    input=prompt,
                    reasoning={"effort": Config.REASONING_EFFORT},
                    text={"verbosity": Config.VERBOSITY}
                )
            )()
            
            import json
            result = json.loads(response.output_text.strip())
            result['document_id'] = document.get('id', 'unknown')
            result['method'] = 'llm'
            return result
            
        except Exception as e:
            print(f"LLM grading failed: {str(e)}")
            return {"grade": "ambiguous", "confidence": 0.5, "reason": "Grading failed", "document_id": document.get('id', 'unknown')}
    
    def _batch_grade_with_cross_encoder(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Grade all documents in a single batch call (fastest method).
        ~50-100ms for 10 documents vs ~4s with LLM.
        """
        cross_encoder = get_cross_encoder()
        if cross_encoder is None or not documents:
            return documents
        
        try:
            # Prepare all query-document pairs
            pairs = [(query, doc.get('text', '')[:512]) for doc in documents]
            
            # Single batch prediction (fast!)
            scores = cross_encoder.predict(pairs)
            
            # Assign grades based on scores
            for i, doc in enumerate(documents):
                score = float(scores[i])
                
                if score > 2.0:
                    grade = "relevant"
                    confidence = min(1.0, (score - 2.0) / 8.0 + 0.7)
                elif score > -1.0:
                    grade = "ambiguous"
                    confidence = 0.5 + (score + 1.0) / 6.0
                else:
                    grade = "irrelevant"
                    confidence = min(1.0, abs(score) / 10.0)
                
                doc['relevance_grade'] = {
                    "grade": grade,
                    "confidence": round(confidence, 3),
                    "score": round(score, 3),
                    "method": "cross_encoder_batch",
                    "document_id": doc.get('id', 'unknown')
                }
            
            return documents
            
        except Exception as e:
            print(f"Batch grading failed: {str(e)}")
            return documents
    
    def retrieve_with_grading(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        CRAG-style retrieval with fast Cross-Encoder grading.
        Uses batch processing for ~20x speedup over LLM grading.
        
        Returns:
            Dict with relevant_docs, ambiguous_docs, needs_web_search flag
        """
        top_k = top_k or Config.TOP_K_RETRIEVAL
        
        # Use RRF hybrid retrieval
        candidates = self.hybrid_retrieve_rrf(query, top_k=top_k * 2)
        docs_to_grade = candidates[:top_k]
        
        # Fast batch grading with Cross-Encoder
        cross_encoder = get_cross_encoder()
        if cross_encoder is not None:
            docs_to_grade = self._batch_grade_with_cross_encoder(query, docs_to_grade)
        else:
            # Fallback to PARALLEL LLM grading
            def grade_single(doc):
                return doc, self.grade_document_relevance(query, doc)
            
            with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                futures = [executor.submit(grade_single, doc) for doc in docs_to_grade]
                for future in as_completed(futures):
                    doc, grade_result = future.result()
                    doc['relevance_grade'] = grade_result
        
        # Categorize documents
        relevant_docs = []
        ambiguous_docs = []
        irrelevant_count = 0
        
        for doc in docs_to_grade:
            grade = doc.get('relevance_grade', {}).get('grade', 'ambiguous')
            
            if grade == 'relevant':
                relevant_docs.append(doc)
            elif grade == 'ambiguous':
                ambiguous_docs.append(doc)
            else:
                irrelevant_count += 1
        
        # If too many irrelevant, suggest web search
        needs_web_search = len(relevant_docs) == 0 and irrelevant_count > top_k // 2
        
        return {
            'relevant_docs': relevant_docs,
            'ambiguous_docs': ambiguous_docs,
            'irrelevant_count': irrelevant_count,
            'needs_web_search': needs_web_search,
            'total_graded': len(relevant_docs) + len(ambiguous_docs) + irrelevant_count
        }
