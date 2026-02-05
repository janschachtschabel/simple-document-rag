"""
Confluence integration for RAG system using LangChain.
Allows searching and loading documents from Confluence spaces.
"""

from typing import List, Dict, Any, Optional
import os
from config import Config

# LangChain Confluence Loader
try:
    from langchain_community.document_loaders import ConfluenceLoader
    CONFLUENCE_AVAILABLE = True
except ImportError:
    CONFLUENCE_AVAILABLE = False
    print("Warning: langchain-community not installed. Run: pip install langchain-community atlassian-python-api")


class ConfluenceRetriever:
    """
    Retriever for Confluence documents.
    Supports both Confluence Cloud and Confluence Server/Data Center.
    """
    
    def __init__(self):
        self.url = os.getenv("CONFLUENCE_URL", "")
        self.username = os.getenv("CONFLUENCE_USERNAME", "")
        self.api_key = os.getenv("CONFLUENCE_API_KEY", "")  # API token for Cloud
        self.space_key = os.getenv("CONFLUENCE_SPACE_KEY", "")
        self.is_cloud = os.getenv("CONFLUENCE_IS_CLOUD", "true").lower() == "true"
        
        self._loader = None
        self._initialized = False
    
    def is_configured(self) -> bool:
        """Check if Confluence is properly configured."""
        return bool(self.url and self.username and self.api_key)
    
    def _get_loader(self) -> Optional[ConfluenceLoader]:
        """Get or create the Confluence loader."""
        if not CONFLUENCE_AVAILABLE:
            print("Confluence loader not available")
            return None
        
        if not self.is_configured():
            print("Confluence not configured. Set CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_KEY")
            return None
        
        if self._loader is None:
            try:
                self._loader = ConfluenceLoader(
                    url=self.url,
                    username=self.username,
                    api_key=self.api_key,
                    cloud=self.is_cloud
                )
                self._initialized = True
            except Exception as e:
                print(f"Failed to initialize Confluence loader: {e}")
                return None
        
        return self._loader
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search Confluence for documents matching the query.
        
        Args:
            query: Search query (CQL or text)
            max_results: Maximum number of results
            
        Returns:
            List of documents with text and metadata
        """
        loader = self._get_loader()
        if not loader:
            return []
        
        try:
            # Use CQL (Confluence Query Language) for search
            cql = f'text ~ "{query}"'
            if self.space_key:
                cql = f'space = "{self.space_key}" AND {cql}'
            
            docs = loader.load(
                cql=cql,
                limit=max_results,
                max_pages=max_results
            )
            
            results = []
            for doc in docs:
                results.append({
                    'text': doc.page_content,
                    'metadata': {
                        'source': doc.metadata.get('source', 'confluence'),
                        'title': doc.metadata.get('title', 'Confluence Page'),
                        'source_type': 'confluence',
                        'page_id': doc.metadata.get('id', ''),
                        'space': doc.metadata.get('space', self.space_key),
                        'url': doc.metadata.get('source', self.url)
                    }
                })
            
            return results
            
        except Exception as e:
            print(f"Confluence search failed: {e}")
            return []
    
    def load_space(self, space_key: str = None, max_pages: int = 100) -> List[Dict[str, Any]]:
        """
        Load all pages from a Confluence space.
        
        Args:
            space_key: Confluence space key (uses configured default if not provided)
            max_pages: Maximum pages to load
            
        Returns:
            List of documents
        """
        loader = self._get_loader()
        if not loader:
            return []
        
        space = space_key or self.space_key
        if not space:
            print("No space key provided")
            return []
        
        try:
            docs = loader.load(
                space_key=space,
                max_pages=max_pages,
                include_attachments=False
            )
            
            results = []
            for doc in docs:
                results.append({
                    'text': doc.page_content,
                    'metadata': {
                        'source': f"confluence:{space}:{doc.metadata.get('title', 'unknown')}",
                        'title': doc.metadata.get('title', 'Confluence Page'),
                        'source_type': 'confluence',
                        'page_id': doc.metadata.get('id', ''),
                        'space': space,
                        'url': doc.metadata.get('source', self.url)
                    }
                })
            
            return results
            
        except Exception as e:
            print(f"Failed to load Confluence space: {e}")
            return []
    
    def load_page(self, page_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific Confluence page by ID."""
        loader = self._get_loader()
        if not loader:
            return None
        
        try:
            docs = loader.load(page_ids=[page_id])
            if docs:
                doc = docs[0]
                return {
                    'text': doc.page_content,
                    'metadata': {
                        'source': f"confluence:{page_id}",
                        'title': doc.metadata.get('title', 'Confluence Page'),
                        'source_type': 'confluence',
                        'page_id': page_id,
                        'url': doc.metadata.get('source', self.url)
                    }
                }
        except Exception as e:
            print(f"Failed to load Confluence page {page_id}: {e}")
        
        return None


# Singleton instance
_confluence_retriever = None

def get_confluence_retriever() -> ConfluenceRetriever:
    """Get or create the Confluence retriever instance."""
    global _confluence_retriever
    if _confluence_retriever is None:
        _confluence_retriever = ConfluenceRetriever()
    return _confluence_retriever


def confluence_available() -> bool:
    """Check if Confluence integration is available and configured."""
    return CONFLUENCE_AVAILABLE and get_confluence_retriever().is_configured()
