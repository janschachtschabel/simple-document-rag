"""
Test suite for the Agentic RAG System

This module contains unit tests for the core components of the RAG system.
"""

import unittest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
import numpy as np

# Import the modules to test
from config import Config
from embeddings import EmbeddingEngine
from vector_db import VectorDatabase
from document_processor import DocumentProcessor
from retriever import SemanticRetriever
from agent import QueryAgent, QueryType
from rag_pipeline import RAGPipeline

class TestConfig(unittest.TestCase):
    """Test configuration module."""
    
    def test_config_values(self):
        """Test that configuration values are properly loaded."""
        self.assertIsInstance(Config.CHUNK_SIZE, int)
        self.assertGreater(Config.CHUNK_SIZE, 0)
        self.assertIsInstance(Config.TOP_K_RETRIEVAL, int)
        self.assertGreater(Config.TOP_K_RETRIEVAL, 0)
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        expected_extensions = {'.txt', '.pdf', '.html', '.htm'}
        self.assertEqual(Config.SUPPORTED_EXTENSIONS, expected_extensions)

class TestEmbeddingEngine(unittest.TestCase):
    """Test embedding engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_engine = EmbeddingEngine()
    
    def test_embedding_dimension(self):
        """Test that embedding dimension is returned correctly."""
        dimension = self.embedding_engine.get_embedding_dimension()
        self.assertIsInstance(dimension, int)
        self.assertGreater(dimension, 0)
    
    def test_text_embedding(self):
        """Test text embedding generation."""
        text = "This is a test sentence."
        embedding = self.embedding_engine.embed_text(text)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding), self.embedding_engine.get_embedding_dimension())
    
    def test_batch_embedding(self):
        """Test batch text embedding."""
        texts = ["First sentence.", "Second sentence."]
        embeddings = self.embedding_engine.embed_text(texts)
        
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), 2)
        for emb in embeddings:
            self.assertIsInstance(emb, np.ndarray)
            self.assertEqual(len(emb), self.embedding_engine.get_embedding_dimension())
    
    def test_similarity_computation(self):
        """Test similarity computation between embeddings."""
        query_text = "artificial intelligence"
        doc_texts = ["machine learning", "natural language processing", "cooking recipes"]
        
        query_embedding = self.embedding_engine.embed_text(query_text)
        doc_embeddings = self.embedding_engine.embed_text(doc_texts)
        
        similarities = self.embedding_engine.compute_similarity(query_embedding, doc_embeddings)
        
        self.assertIsInstance(similarities, list)
        self.assertEqual(len(similarities), 3)
        for sim in similarities:
            self.assertIsInstance(sim, float)
            self.assertGreaterEqual(sim, -1.0)
            self.assertLessEqual(sim, 1.0)

class TestDocumentProcessor(unittest.TestCase):
    """Test document processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()
    
    def test_text_processing(self):
        """Test direct text processing."""
        text = "This is a test document for processing."
        metadata = {"topic": "test"}
        
        documents = self.processor.process_text_direct(text, metadata)
        
        self.assertIsInstance(documents, list)
        self.assertGreater(len(documents), 0)
        
        for doc in documents:
            self.assertIn('text', doc)
            self.assertIn('metadata', doc)
            self.assertEqual(doc['metadata']['topic'], 'test')
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        dirty_text = "  This   is    a   test   with   extra   spaces.  "
        clean_text = self.processor._clean_text(dirty_text)
        
        self.assertEqual(clean_text, "This is a test with extra spaces.")
    
    def test_text_chunking(self):
        """Test text chunking."""
        long_text = " ".join(["word"] * 200)  # Create a long text
        chunks = self.processor._chunk_text(long_text)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)
        
        # Check that chunks don't exceed chunk size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), self.processor.chunk_size + 50)  # Allow some tolerance
    
    def test_txt_processing(self):
        """Test TXT file processing."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test text file.\nIt has multiple lines.\nAnd some content.")
            temp_file = f.name
        
        try:
            documents = self.processor.process_file(temp_file)
            
            self.assertIsInstance(documents, list)
            self.assertGreater(len(documents), 0)
            
            for doc in documents:
                self.assertIn('text', doc)
                self.assertIn('metadata', doc)
                self.assertEqual(doc['metadata']['file_type'], 'txt')
                
        finally:
            os.unlink(temp_file)
    
    def test_unsupported_file_type(self):
        """Test handling of unsupported file types."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_file = f.name
        
        try:
            with self.assertRaises(ValueError):
                self.processor.process_file(temp_file)
        finally:
            os.unlink(temp_file)

class TestVectorDatabase(unittest.TestCase):
    """Test vector database."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.vector_db = VectorDatabase(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_documents(self):
        """Test adding documents to the database."""
        documents = [
            {
                'text': 'First document about AI',
                'metadata': {'topic': 'AI', 'source': 'test'}
            },
            {
                'text': 'Second document about ML',
                'metadata': {'topic': 'ML', 'source': 'test'}
            }
        ]
        
        doc_ids = self.vector_db.add_documents(documents)
        
        self.assertIsInstance(doc_ids, list)
        self.assertEqual(len(doc_ids), 2)
        
        for doc_id in doc_ids:
            self.assertIsInstance(doc_id, str)
    
    def test_query_documents(self):
        """Test querying documents."""
        # Add test documents
        documents = [
            {
                'text': 'Artificial intelligence and machine learning',
                'metadata': {'topic': 'AI'}
            }
        ]
        
        self.vector_db.add_documents(documents)
        
        # Query
        results = self.vector_db.query('machine learning', n_results=1)
        
        self.assertIn('results', results)
        self.assertGreater(len(results['results']), 0)
        
        result = results['results'][0]
        self.assertIn('id', result)
        self.assertIn('text', result)
        self.assertIn('metadata', result)
    
    def test_get_document_by_id(self):
        """Test retrieving document by ID."""
        documents = [
            {
                'text': 'Test document for ID retrieval',
                'metadata': {'topic': 'test'}
            }
        ]
        
        doc_ids = self.vector_db.add_documents(documents)
        doc_id = doc_ids[0]
        
        retrieved_doc = self.vector_db.get_document_by_id(doc_id)
        
        self.assertIsNotNone(retrieved_doc)
        self.assertEqual(retrieved_doc['id'], doc_id)
        self.assertEqual(retrieved_doc['text'], 'Test document for ID retrieval')
    
    def test_delete_document(self):
        """Test deleting a document."""
        documents = [
            {
                'text': 'Document to be deleted',
                'metadata': {'topic': 'test'}
            }
        ]
        
        doc_ids = self.vector_db.add_documents(documents)
        doc_id = doc_ids[0]
        
        # Delete the document
        success = self.vector_db.delete_document(doc_id)
        self.assertTrue(success)
        
        # Verify it's deleted
        retrieved_doc = self.vector_db.get_document_by_id(doc_id)
        self.assertIsNone(retrieved_doc)
    
    def test_collection_stats(self):
        """Test getting collection statistics."""
        stats = self.vector_db.get_collection_stats()
        
        self.assertIn('total_documents', stats)
        self.assertIn('collection_name', stats)
        self.assertIn('persist_directory', stats)
        
        self.assertIsInstance(stats['total_documents'], int)
        self.assertGreaterEqual(stats['total_documents'], 0)

class TestSemanticRetriever(unittest.TestCase):
    """Test semantic retriever."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_db = VectorDatabase(self.temp_dir)
        self.retriever = SemanticRetriever(self.vector_db)
        
        # Add test documents
        documents = [
            {
                'text': 'Artificial intelligence is transforming technology',
                'metadata': {'topic': 'AI', 'source': 'test1'}
            },
            {
                'text': 'Machine learning algorithms learn from data',
                'metadata': {'topic': 'ML', 'source': 'test2'}
            },
            {
                'text': 'Deep learning uses neural networks',
                'metadata': {'topic': 'DL', 'source': 'test3'}
            }
        ]
        
        self.vector_db.add_documents(documents)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_retrieve_documents(self):
        """Test document retrieval."""
        results = self.retriever.retrieve('artificial intelligence', top_k=2)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
        
        for result in results:
            self.assertIn('id', result)
            self.assertIn('text', result)
            self.assertIn('metadata', result)
            self.assertIn('similarity_score', result)
    
    def test_retrieve_with_filters(self):
        """Test document retrieval with filters."""
        filters = {'topic': 'AI'}
        results = self.retriever.retrieve('intelligence', filters=filters)
        
        for result in results:
            self.assertEqual(result['metadata']['topic'], 'AI')
    
    def test_hybrid_retrieve(self):
        """Test hybrid retrieval."""
        results = self.retriever.hybrid_retrieve('learning', top_k=2)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
        
        for result in results:
            self.assertIn('combined_score', result)

class TestQueryAgent(unittest.TestCase):
    """Test query agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_db = VectorDatabase(self.temp_dir)
        self.retriever = SemanticRetriever(self.vector_db)
        self.agent = QueryAgent(self.retriever)
        
        # Add test documents
        documents = [
            {
                'text': 'Python is a popular programming language for AI development',
                'metadata': {'topic': 'Python', 'source': 'test'}
            }
        ]
        
        self.vector_db.add_documents(documents)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_query_analysis(self):
        """Test query analysis."""
        query = "What is Python programming?"
        analysis = self.agent._analyze_query(query)
        
        self.assertIn('query_type', analysis)
        self.assertIn('entities', analysis)
        self.assertIn('keywords', analysis)
        
        self.assertIsInstance(analysis['query_type'], str)
        self.assertIsInstance(analysis['entities'], list)
        self.assertIsInstance(analysis['keywords'], list)
    
    def test_fallback_query_analysis(self):
        """Test fallback query analysis."""
        query = "How does machine learning work?"
        analysis = self.agent._fallback_query_analysis(query)
        
        self.assertIn('query_type', analysis)
        self.assertIn('entities', analysis)
        self.assertIn('keywords', analysis)
    
    def test_context_preparation(self):
        """Test context preparation from documents."""
        documents = [
            {
                'id': 'doc1',
                'text': 'First document content',
                'metadata': {'source': 'test1'}
            },
            {
                'id': 'doc2',
                'text': 'Second document content',
                'metadata': {'source': 'test2'}
            }
        ]
        
        context = self.agent._prepare_context(documents)
        
        self.assertIsInstance(context, str)
        self.assertIn('First document content', context)
        self.assertIn('Second document content', context)
        self.assertIn('Source 1', context)
        self.assertIn('Source 2', context)
    
    def test_conversation_history(self):
        """Test conversation history management."""
        # Initially empty
        history = self.agent.get_conversation_history()
        self.assertEqual(len(history), 0)
        
        # Clear history (should not raise error)
        self.agent.clear_history()
        history = self.agent.get_conversation_history()
        self.assertEqual(len(history), 0)

class TestRAGPipeline(unittest.TestCase):
    """Test RAG pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = RAGPipeline()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_text_document(self):
        """Test adding text document."""
        result = self.pipeline.add_document(
            {"text": "Test document content"},
            {"topic": "test"}
        )
        
        self.assertTrue(result['success'])
        self.assertIn('document_ids', result)
        self.assertGreater(result['chunks_processed'], 0)
    
    def test_query_system(self):
        """Test querying the system."""
        # Add a document first
        self.pipeline.add_document(
            {"text": "Artificial intelligence is the future of technology"},
            {"topic": "AI"}
        )
        
        # Query
        result = self.pipeline.query("What is artificial intelligence?")
        
        self.assertIn('answer', result)
        self.assertIn('question', result)
        self.assertIn('retrieval_strategy', result)
        self.assertIn('query_analysis', result)
    
    def test_search_documents(self):
        """Test document search."""
        # Add documents
        self.pipeline.add_document(
            {"text": "Machine learning algorithms"},
            {"topic": "ML"}
        )
        
        # Search
        result = self.pipeline.search_documents("machine learning")
        
        self.assertIn('results', result)
        self.assertIn('query', result)
        self.assertIn('total_results', result)
    
    def test_get_stats(self):
        """Test getting system statistics."""
        stats = self.pipeline.get_stats()
        
        self.assertIn('database', stats)
        self.assertIn('retrieval_methods', stats)
        self.assertIn('supported_formats', stats)
        self.assertIn('chunk_size', stats)
    
    def test_health_check(self):
        """Test health check."""
        health = self.pipeline.health_check()
        
        self.assertIn('status', health)
        self.assertIn('components', health)
        self.assertIn('timestamp', health)
    
    def test_explain_query(self):
        """Test query explanation."""
        explanation = self.pipeline.explain_query("What is AI?")
        
        self.assertIn('query', explanation)
        self.assertIn('analysis', explanation)
        self.assertIn('processing_steps', explanation)

def run_tests():
    """Run all tests."""
    print("🧪 Running Agentic RAG System Tests")
    print("=" * 50)
    
    # Create test suite
    test_classes = [
        TestConfig,
        TestEmbeddingEngine,
        TestDocumentProcessor,
        TestVectorDatabase,
        TestSemanticRetriever,
        TestQueryAgent,
        TestRAGPipeline
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_tests()
