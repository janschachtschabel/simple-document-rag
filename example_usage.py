"""
Example usage of the Agentic RAG System

This script demonstrates how to use the RAG pipeline programmatically.
"""

import os
import sys
from rag_pipeline import RAGPipeline
from config import Config

def main():
    """Main example function."""
    print("🤖 Agentic RAG System - Example Usage")
    print("=" * 50)
    
    # Initialize the RAG pipeline
    print("\n📚 Initializing RAG Pipeline...")
    rag = RAGPipeline()
    
    # Health check
    print("\n🔍 Performing health check...")
    health = rag.health_check()
    print(f"Status: {health['status']}")
    if health['status'] == 'healthy':
        print("✅ All components operational")
    else:
        print(f"❌ Issues found: {health.get('error', 'Unknown')}")
        return
    
    # Example 1: Add text documents
    print("\n📝 Example 1: Adding text documents")
    print("-" * 30)
    
    sample_documents = [
        {
            "text": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.",
            "metadata": {"topic": "AI", "type": "definition"}
        },
        {
            "text": "Machine Learning is a subset of AI that focuses on the development of algorithms that can learn from and make predictions or decisions based on data. It enables computers to improve their performance on a task through experience without being explicitly programmed.",
            "metadata": {"topic": "Machine Learning", "type": "definition"}
        },
        {
            "text": "Deep Learning is a subfield of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in areas such as image recognition, natural language processing, and speech recognition.",
            "metadata": {"topic": "Deep Learning", "type": "definition"}
        }
    ]
    
    for i, doc in enumerate(sample_documents, 1):
        print(f"Adding document {i}: {doc['metadata']['topic']}")
        result = rag.add_document(doc, doc['metadata'])
        if result['success']:
            print(f"✅ {result['message']}")
        else:
            print(f"❌ Error: {result.get('error')}")
    
    # Example 2: Query the system
    print("\n❓ Example 2: Querying the system")
    print("-" * 30)
    
    queries = [
        "What is Artificial Intelligence?",
        "How does Machine Learning work?",
        "Compare AI and Machine Learning",
        "What are the applications of Deep Learning?",
        "Explain the relationship between AI, ML, and Deep Learning"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        
        # Try different retrieval strategies
        strategies = ["semantic", "hybrid", "rerank"]
        
        for strategy in strategies:
            print(f"  Using {strategy} retrieval...")
            result = rag.query(query, retrieval_strategy=strategy, top_k=3)
            
            if "answer" in result:
                print(f"  ✅ Answer: {result['answer'][:100]}...")
                print(f"  📊 Retrieved {result['documents_retrieved']} documents")
                print(f"  🏷️ Query type: {result['query_analysis']['query_type']}")
            else:
                print(f"  ❌ Error: {result.get('error')}")
        
        print("-" * 50)
    
    # Example 3: Search documents
    print("\n🔍 Example 3: Searching documents")
    print("-" * 30)
    
    search_query = "neural networks"
    print(f"Searching for: {search_query}")
    
    search_results = rag.search_documents(search_query, top_k=5)
    
    if "results" in search_results:
        print(f"Found {len(search_results['results'])} results:")
        for i, result in enumerate(search_results['results'], 1):
            print(f"  {i}. Score: {result['similarity_score']:.3f}")
            print(f"     Preview: {result['text'][:100]}...")
            print(f"     Source: {result['metadata'].get('topic', 'Unknown')}")
    else:
        print(f"❌ Search failed: {search_results.get('error')}")
    
    # Example 4: Get system statistics
    print("\n📊 Example 4: System statistics")
    print("-" * 30)
    
    stats = rag.get_stats()
    print(f"Database stats: {stats['database']}")
    print(f"Supported formats: {stats['supported_formats']}")
    print(f"Chunk size: {stats['chunk_size']}")
    print(f"Top K default: {stats['top_k_default']}")
    
    # Example 5: Explain query processing
    print("\n🔍 Example 5: Query explanation")
    print("-" * 30)
    
    explain_query = "What are the differences between supervised and unsupervised learning?"
    print(f"Explaining query: {explain_query}")
    
    explanation = rag.explain_query(explain_query)
    
    if "analysis" in explanation:
        print(f"Query type: {explanation['analysis']['query_type']}")
        print(f"Entities: {explanation['analysis']['entities']}")
        print(f"Complexity: {explanation['analysis']['complexity']}")
        print(f"Processing steps: {explanation['processing_steps']}")
    else:
        print(f"❌ Explanation failed: {explanation.get('error')}")
    
    # Example 6: Conversation history
    print("\n💬 Example 6: Conversation history")
    print("-" * 30)
    
    history = rag.get_conversation_history()
    print(f"Total interactions: {len(history)}")
    
    if history:
        print("Recent interactions:")
        for i, interaction in enumerate(history[-3:], 1):
            print(f"  {i}. Q: {interaction['query'][:50]}...")
            print(f"     A: {interaction['response'][:50]}...")
    
    print("\n🎉 Example usage completed!")
    print("=" * 50)

def demo_with_file_upload():
    """Demonstrate file upload functionality."""
    print("\n📁 Demo: File Upload")
    print("-" * 30)
    
    # Create a sample text file
    sample_file = "sample_ai_document.txt"
    sample_content = """
    Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret, and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit of filling the gap between human communication and computer understanding.
    
    Key components of NLP include:
    1. Speech Recognition - Converting spoken words into text
    2. Natural Language Understanding - Comprehending the meaning of text
    3. Natural Language Generation - Producing human-like text
    4. Machine Translation - Translating text between languages
    
    Modern NLP systems use deep learning techniques, particularly transformer models like BERT and GPT, which have revolutionized the field by enabling more sophisticated understanding and generation of human language.
    """
    
    try:
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        
        print(f"Created sample file: {sample_file}")
        
        # Initialize RAG pipeline
        rag = RAGPipeline()
        
        # Add the file
        print("Uploading file to RAG system...")
        result = rag.add_document(sample_file, {"topic": "NLP", "source_type": "file"})
        
        if result['success']:
            print(f"✅ {result['message']}")
            
            # Query about the content
            query = "What is Natural Language Processing?"
            print(f"\nQuerying: {query}")
            
            response = rag.query(query)
            if "answer" in response:
                print(f"Answer: {response['answer']}")
                print(f"Sources: {len(response.get('sources', []))}")
        else:
            print(f"❌ Error: {result.get('error')}")
        
    finally:
        # Clean up
        if os.path.exists(sample_file):
            os.remove(sample_file)
            print(f"Cleaned up sample file: {sample_file}")

def demo_with_url():
    """Demonstrate URL processing functionality."""
    print("\n🌐 Demo: URL Processing")
    print("-" * 30)
    
    # Example URL (using a simple, accessible URL)
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    print(f"Processing URL: {url}")
    
    try:
        rag = RAGPipeline()
        
        # Add URL
        result = rag.add_document(url, {"topic": "AI", "source_type": "web"})
        
        if result['success']:
            print(f"✅ {result['message']}")
            
            # Query about the content
            query = "What are the main applications of artificial intelligence?"
            print(f"\nQuerying: {query}")
            
            response = rag.query(query, retrieval_strategy="hybrid")
            if "answer" in response:
                print(f"Answer: {response['answer'][:200]}...")
                print(f"Retrieval strategy: {response['retrieval_strategy']}")
        else:
            print(f"❌ Error: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ URL processing failed: {str(e)}")

if __name__ == "__main__":
    try:
        # Check if OpenAI API key is set
        if not Config.OPENAI_API_KEY:
            print("❌ Error: OPENAI_API_KEY is not set!")
            print("Please set your OpenAI API key in the environment variables or .env file.")
            sys.exit(1)
        
        # Run main examples
        main()
        
        # Ask user if they want to run additional demos
        print("\n" + "=" * 50)
        choice = input("Would you like to run additional demos? (y/n): ").lower().strip()
        
        if choice == 'y':
            demo_with_file_upload()
            
            choice = input("\nTry URL processing demo? (y/n): ").lower().strip()
            if choice == 'y':
                demo_with_url()
        
        print("\n👋 Thanks for trying the Agentic RAG System!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
