"""
Confluence integration for RAG system using Atlassian API directly.
Allows searching and loading documents from Confluence spaces.
"""

from typing import List, Dict, Any, Optional
import os
from config import Config

# Try Atlassian API directly (more reliable than LangChain wrapper)
try:
    from atlassian import Confluence
    CONFLUENCE_AVAILABLE = True
except ImportError:
    CONFLUENCE_AVAILABLE = False
    print("Warning: atlassian-python-api not installed. Run: pip install atlassian-python-api")


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
    
    def _get_client(self) -> Optional[Confluence]:
        """Get or create the Confluence API client."""
        if not CONFLUENCE_AVAILABLE:
            print("Confluence API not available - install atlassian-python-api")
            return None
        
        if not self.is_configured():
            print("Confluence not configured. Set CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_KEY")
            return None
        
        if self._loader is None:
            try:
                self._loader = Confluence(
                    url=self.url,
                    username=self.username,
                    password=self.api_key,  # API token is used as password
                    cloud=self.is_cloud
                )
                self._initialized = True
                print(f"Confluence client initialized for {self.url}")
            except Exception as e:
                print(f"Failed to initialize Confluence client: {e}")
                return None
        
        return self._loader
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search Confluence for documents matching the query using Atlassian API.
        
        Args:
            query: Search query text
            max_results: Maximum number of results
            
        Returns:
            List of documents with text and metadata
        """
        client = self._get_client()
        if not client:
            return []
        
        try:
            # Use CQL (Confluence Query Language) for search
            cql = f'text ~ "{query}"'
            if self.space_key:
                cql = f'space = "{self.space_key}" AND {cql}'
            
            # Search using Atlassian API directly
            search_results = client.cql(cql, limit=max_results)
            
            results = []
            pages = search_results.get('results', [])
            
            for page in pages:
                try:
                    # Get page content
                    content = page.get('content', {})
                    page_id = content.get('id', page.get('id', ''))
                    title = content.get('title', page.get('title', 'Confluence Page'))
                    
                    # Get the page body if available
                    page_text = ""
                    if page_id:
                        try:
                            page_content = client.get_page_by_id(page_id, expand='body.storage')
                            body = page_content.get('body', {}).get('storage', {}).get('value', '')
                            # Strip HTML tags for plain text
                            import re
                            page_text = re.sub('<[^<]+?>', ' ', body)
                            page_text = ' '.join(page_text.split())[:2000]  # Limit length
                        except Exception as e:
                            print(f"Could not get page content for {page_id}: {e}")
                            page_text = page.get('excerpt', title)
                    else:
                        page_text = page.get('excerpt', title)
                    
                    # Build page URL
                    space_key = content.get('space', {}).get('key', self.space_key) if isinstance(content.get('space'), dict) else self.space_key
                    page_url = f"{self.url}/wiki/spaces/{space_key}/pages/{page_id}" if page_id else self.url
                    
                    results.append({
                        'text': page_text,
                        'metadata': {
                            'source': f"confluence:{title}",
                            'title': title,
                            'source_type': 'confluence',
                            'page_id': str(page_id),
                            'space': space_key,
                            'url': page_url
                        }
                    })
                except Exception as e:
                    print(f"Error processing Confluence page: {e}")
                    continue
            
            print(f"Confluence search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            import traceback
            print(f"Confluence search failed: {e}")
            print(traceback.format_exc())
            return []
    
    def load_space(self, space_key: str = None, max_pages: int = 100) -> List[Dict[str, Any]]:
        """
        Load all pages from a Confluence space using Atlassian API.
        
        Args:
            space_key: Confluence space key (uses configured default if not provided)
            max_pages: Maximum pages to load
            
        Returns:
            List of documents
        """
        client = self._get_client()
        if not client:
            return []
        
        space = space_key or self.space_key
        if not space:
            print("No space key provided")
            return []
        
        try:
            import re
            pages = client.get_all_pages_from_space(space, limit=max_pages, expand='body.storage')
            
            results = []
            for page in pages:
                body = page.get('body', {}).get('storage', {}).get('value', '')
                page_text = re.sub('<[^<]+?>', ' ', body)
                page_text = ' '.join(page_text.split())[:2000]
                
                results.append({
                    'text': page_text,
                    'metadata': {
                        'source': f"confluence:{space}:{page.get('title', 'unknown')}",
                        'title': page.get('title', 'Confluence Page'),
                        'source_type': 'confluence',
                        'page_id': page.get('id', ''),
                        'space': space,
                        'url': f"{self.url}/wiki/spaces/{space}/pages/{page.get('id', '')}"
                    }
                })
            
            return results
            
        except Exception as e:
            print(f"Failed to load Confluence space: {e}")
            return []
    
    def load_page(self, page_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific Confluence page by ID using Atlassian API."""
        client = self._get_client()
        if not client:
            return None
        
        try:
            import re
            page = client.get_page_by_id(page_id, expand='body.storage')
            body = page.get('body', {}).get('storage', {}).get('value', '')
            page_text = re.sub('<[^<]+?>', ' ', body)
            page_text = ' '.join(page_text.split())
            
            return {
                'text': page_text,
                'metadata': {
                    'source': f"confluence:{page_id}",
                    'title': page.get('title', 'Confluence Page'),
                    'source_type': 'confluence',
                    'page_id': page_id,
                    'url': f"{self.url}/wiki/pages/{page_id}"
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
