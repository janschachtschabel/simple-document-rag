import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import uuid
from embeddings import EmbeddingEngine
from config import Config

class VectorDatabase:
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or Config.CHROMA_PERSIST_DIRECTORY
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = None
        self.embedding_engine = EmbeddingEngine()
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get the collection."""
        try:
            self.collection = self.client.get_collection(name="documents")
        except:
            self.collection = self.client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of dictionaries containing 'text', 'metadata', and optionally 'id'
        
        Returns:
            List of document IDs
        """
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            doc_id = doc.get('id', str(uuid.uuid4()))
            ids.append(doc_id)
            texts.append(doc['text'])
            # Filter out None values from metadata (ChromaDB doesn't accept None)
            raw_metadata = doc.get('metadata', {})
            clean_metadata = {k: v for k, v in raw_metadata.items() if v is not None}
            metadatas.append(clean_metadata)
        
        # Generate embeddings
        embeddings = self.embedding_engine.embed_text(texts)
        
        # Convert embeddings to list format for ChromaDB
        if isinstance(embeddings, list):
            # List of numpy arrays - convert each to list
            embeddings_list = [emb.tolist() for emb in embeddings]
        else:
            # Single numpy array
            embeddings_list = [embeddings.tolist()]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings_list
        )
        
        return ids
    
    def query(self, query_text: str, n_results: int = None) -> Dict[str, Any]:
        """
        Query the vector database for similar documents.
        
        Args:
            query_text: The query text
            n_results: Number of results to return
        
        Returns:
            Dictionary containing results
        """
        n_results = n_results or Config.TOP_K_RETRIEVAL
        
        try:
            # Generate query embedding using the same engine as document embeddings
            query_embedding = self.embedding_engine.embed_text(query_text)
            query_embedding_list = query_embedding.tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            return {
                'results': formatted_results,
                'query': query_text,
                'total_results': len(formatted_results)
            }
            
        except Exception as e:
            raise Exception(f"Query error: {str(e)}")
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by its ID."""
        try:
            results = self.collection.get(ids=[doc_id])
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'text': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            return None
        except Exception as e:
            raise Exception(f"Get document error: {str(e)}")
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by its ID."""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            raise Exception(f"Delete document error: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': 'documents',
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            raise Exception(f"Get stats error: {str(e)}")
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(name="documents")
            self._initialize_collection()
        except Exception as e:
            raise Exception(f"Clear collection error: {str(e)}")
    
    def list_sources(self) -> List[Dict[str, Any]]:
        """
        List all unique sources/documents in the database.
        Returns aggregated info per source file.
        """
        try:
            # Get all documents with metadata
            all_docs = self.collection.get(include=['metadatas'])
            
            if not all_docs['ids']:
                return []
            
            # Aggregate by source
            sources = {}
            for i, doc_id in enumerate(all_docs['ids']):
                metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                source = metadata.get('source', metadata.get('original_filename', 'Unknown'))
                
                if source not in sources:
                    sources[source] = {
                        'source': source,
                        'chunk_count': 0,
                        'chunk_ids': [],
                        'metadata': metadata
                    }
                
                sources[source]['chunk_count'] += 1
                sources[source]['chunk_ids'].append(doc_id)
            
            return list(sources.values())
            
        except Exception as e:
            raise Exception(f"List sources error: {str(e)}")
    
    def delete_source(self, source_name: str) -> Dict[str, Any]:
        """
        Delete all chunks belonging to a specific source.
        
        Args:
            source_name: Name of the source file to delete
            
        Returns:
            Dict with deletion stats
        """
        try:
            # Find all chunks with this source
            all_docs = self.collection.get(include=['metadatas'])
            
            chunks_to_delete = []
            for i, doc_id in enumerate(all_docs['ids']):
                metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                source = metadata.get('source', metadata.get('original_filename', ''))
                
                if source == source_name:
                    chunks_to_delete.append(doc_id)
            
            if not chunks_to_delete:
                return {
                    'success': False,
                    'message': f"Source '{source_name}' not found",
                    'deleted_chunks': 0
                }
            
            # Delete all chunks
            self.collection.delete(ids=chunks_to_delete)
            
            return {
                'success': True,
                'message': f"Deleted source '{source_name}'",
                'deleted_chunks': len(chunks_to_delete)
            }
            
        except Exception as e:
            raise Exception(f"Delete source error: {str(e)}")
