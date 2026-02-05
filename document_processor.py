import os
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import re
from config import Config

# Import markitdown for universal document conversion
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    print("Warning: markitdown not available. Install with: pip install 'markitdown[all]'")

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
        
        # Initialize markitdown converter
        if MARKITDOWN_AVAILABLE:
            self.md_converter = MarkItDown(enable_plugins=False)
        else:
            self.md_converter = None
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a file and return chunks with metadata.
        Uses markitdown to convert all supported formats to markdown first.
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        
        # Check if file type is supported
        if file_ext not in Config.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_ext}. Supported: {Config.SUPPORTED_EXTENSIONS}")
        
        # Use markitdown for conversion if available
        if MARKITDOWN_AVAILABLE and self.md_converter:
            return self._process_with_markitdown(file_path, file_ext, filename)
        
        # Fallback to legacy processing for basic formats
        if file_ext == '.txt':
            return self._process_txt(file_path)
        elif file_ext in ['.html', '.htm']:
            return self._process_html_legacy(file_path)
        else:
            raise ValueError(f"markitdown not available. Cannot process {file_ext} files.")
    
    def _process_with_markitdown(self, file_path: str, file_ext: str, filename: str) -> List[Dict[str, Any]]:
        """
        Process any supported file using markitdown.
        Converts to markdown first, then chunks.
        """
        try:
            # Convert to markdown using markitdown
            result = self.md_converter.convert(file_path)
            markdown_text = result.text_content
            
            if not markdown_text or not markdown_text.strip():
                raise ValueError(f"No text content extracted from {filename}")
            
            # Clean the text
            text = self._clean_text(markdown_text)
            
            # Chunk the text
            chunks = self._chunk_text(text)
            
            # Determine file type category
            file_type = self._get_file_type_category(file_ext)
            
            # Create metadata
            metadata = {
                'source': filename,
                'source_type': 'file',
                'filename': filename,
                'file_type': file_type,
                'original_format': file_ext.lstrip('.'),
                'converted_via': 'markitdown'
            }
            
            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata['chunk_index'] = i
                doc_metadata['total_chunks'] = len(chunks)
                
                documents.append({
                    'text': chunk,
                    'metadata': doc_metadata
                })
            
            return documents
            
        except Exception as e:
            raise Exception(f"Error processing {filename} with markitdown: {str(e)}")
    
    def _get_file_type_category(self, file_ext: str) -> str:
        """Get a category name for the file type."""
        categories = {
            '.pdf': 'pdf',
            '.docx': 'word', '.doc': 'word',
            '.pptx': 'powerpoint', '.ppt': 'powerpoint',
            '.xlsx': 'excel', '.xls': 'excel',
            '.html': 'html', '.htm': 'html',
            '.txt': 'text', '.md': 'markdown', '.markdown': 'markdown',
            '.csv': 'csv', '.json': 'json', '.xml': 'xml',
            '.epub': 'epub',
            '.msg': 'email', '.eml': 'email',
            '.jpg': 'image', '.jpeg': 'image', '.png': 'image', 
            '.gif': 'image', '.bmp': 'image', '.tiff': 'image', '.webp': 'image',
            '.zip': 'archive',
            '.rst': 'text'
        }
        return categories.get(file_ext.lower(), 'unknown')
    
    def process_url(self, url: str) -> List[Dict[str, Any]]:
        """Process a web page and return chunks with metadata."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            text = self._clean_text(text)
            
            # Create metadata
            metadata = {
                'source': url,
                'source_type': 'web',
                'title': soup.title.string if soup.title else url,
                'description': soup.find('meta', attrs={'name': 'description'}).get('content', '') if soup.find('meta', attrs={'name': 'description'}) else ''
            }
            
            # Chunk the text
            chunks = self._chunk_text(text)
            
            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata['chunk_index'] = i
                doc_metadata['total_chunks'] = len(chunks)
                
                documents.append({
                    'text': chunk,
                    'metadata': doc_metadata
                })
            
            return documents
            
        except Exception as e:
            raise Exception(f"Error processing URL {url}: {str(e)}")
    
    def _process_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            text = self._clean_text(text)
            chunks = self._chunk_text(text)
            
            metadata = {
                'source': file_path,
                'source_type': 'file',
                'filename': os.path.basename(file_path),
                'file_type': 'txt'
            }
            
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata['chunk_index'] = i
                doc_metadata['total_chunks'] = len(chunks)
                
                documents.append({
                    'text': chunk,
                    'metadata': doc_metadata
                })
            
            return documents
            
        except Exception as e:
            raise Exception(f"Error processing TXT file {file_path}: {str(e)}")
    
    def _process_html_legacy(self, file_path: str) -> List[Dict[str, Any]]:
        """Process an HTML file (legacy fallback when markitdown not available)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            text = self._clean_text(text)
            
            # Chunk the text
            chunks = self._chunk_text(text)
            
            metadata = {
                'source': file_path,
                'source_type': 'file',
                'filename': os.path.basename(file_path),
                'file_type': 'html',
                'title': soup.title.string if soup.title else os.path.basename(file_path)
            }
            
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata['chunk_index'] = i
                doc_metadata['total_chunks'] = len(chunks)
                
                documents.append({
                    'text': chunk,
                    'metadata': doc_metadata
                })
            
            return documents
            
        except Exception as e:
            raise Exception(f"Error processing HTML file {file_path}: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text while preserving important characters."""
        # Remove excessive whitespace (but preserve single spaces)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Only remove truly problematic control characters, keep all printable unicode
        # This preserves German umlauts (äöüÄÖÜß), accents, and other international characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of specified size with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this is the last chunk, take everything remaining
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at a sentence boundary
            chunk = text[start:end]
            
            # Look for sentence endings near the chunk boundary
            sentence_endings = ['.', '!', '?', '\n']
            best_break = -1
            
            for i in range(len(chunk) - 1, max(0, len(chunk) - 200), -1):
                if chunk[i] in sentence_endings:
                    best_break = i + 1
                    break
            
            if best_break > 0:
                chunks.append(chunk[:best_break])
                start = start + best_break - self.chunk_overlap
            else:
                chunks.append(chunk)
                start = end - self.chunk_overlap
        
        # Remove empty chunks and whitespace-only chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        return chunks
    
    def process_text_direct(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Process raw text directly."""
        text = self._clean_text(text)
        chunks = self._chunk_text(text)
        
        base_metadata = metadata or {
            'source': 'direct_input',
            'source_type': 'text'
        }
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = base_metadata.copy()
            doc_metadata['chunk_index'] = i
            doc_metadata['total_chunks'] = len(chunks)
            
            documents.append({
                'text': chunk,
                'metadata': doc_metadata
            })
        
        return documents
