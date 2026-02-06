from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import Config

# Batch size for parallel embedding (OpenAI limit is ~2048 texts per call)
EMBEDDING_BATCH_SIZE = 100

class EmbeddingEngine:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self._model = None
        self._client = None
        self._initialize_model()
    
    def _initialize_model(self):
        if self.model_name.startswith("text-embedding"):
            # OpenAI embedding model
            self.model_type = "openai"
            self._client = OpenAI(api_key=Config.OPENAI_API_KEY)
        else:
            # Sentence transformer model
            self.model_type = "sentence_transformer"
            self._model = SentenceTransformer(self.model_name)
    
    def embed_text(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate embeddings for text or list of texts."""
        if self.model_type == "openai":
            return self._embed_openai(text)
        else:
            return self._embed_sentence_transformer(text)
    
    def _embed_openai(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate embeddings using OpenAI API v1.0+ with parallel batch processing."""
        single_input = isinstance(text, str)
        if single_input:
            text = [text]
        
        # For small batches, single API call
        if len(text) <= EMBEDDING_BATCH_SIZE:
            embeddings = self._embed_batch(text)
            return embeddings[0] if single_input else embeddings
        
        # For large batches, parallel processing
        batches = [text[i:i + EMBEDDING_BATCH_SIZE] for i in range(0, len(text), EMBEDDING_BATCH_SIZE)]
        all_embeddings = [None] * len(batches)
        
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            future_to_idx = {executor.submit(self._embed_batch, batch): i for i, batch in enumerate(batches)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                all_embeddings[idx] = future.result()
        
        # Flatten results maintaining order
        result = []
        for batch_embeddings in all_embeddings:
            if isinstance(batch_embeddings, list):
                result.extend(batch_embeddings)
            else:
                result.append(batch_embeddings)
        
        return result[0] if single_input else result
    
    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a single batch of texts."""
        try:
            response = self._client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return [np.array(item.embedding) for item in response.data]
        except Exception as e:
            raise Exception(f"OpenAI embedding error: {str(e)}")
    
    def _embed_sentence_transformer(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate embeddings using sentence transformers."""
        if isinstance(text, str):
            return self._model.encode(text)
        else:
            return self._model.encode(text)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        if self.model_type == "openai":
            # OpenAI text-embedding-ada-002 returns 1536 dimensions
            return 1536
        else:
            return self._model.get_sentence_embedding_dimension()
    
    def compute_similarity(self, query_embedding: np.ndarray, document_embeddings: List[np.ndarray]) -> List[float]:
        """Compute cosine similarity between query and documents."""
        similarities = []
        for doc_embedding in document_embeddings:
            # Normalize vectors
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norm = doc_embedding / np.linalg.norm(doc_embedding)
            
            # Compute cosine similarity
            similarity = np.dot(query_norm, doc_norm)
            similarities.append(float(similarity))
        
        return similarities
