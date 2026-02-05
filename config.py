import os
from dotenv import load_dotenv

# Load from .env file if it exists (OS environment variables take precedence)
load_dotenv(override=False)

class Config:
    # OpenAI Configuration - loaded from OS environment variables or .env file
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    # Document Processing
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "30000"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval Configuration
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    
    # Supported Document Types (via markitdown)
    SUPPORTED_EXTENSIONS = {
        # Text formats
        '.txt', '.md', '.markdown', '.rst', '.csv', '.json', '.xml',
        # Office documents
        '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls',
        # Web formats
        '.html', '.htm',
        # Other formats
        '.epub', '.msg', '.eml',
        # Images (OCR)
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
        # Archives
        '.zip'
    }
    
    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your environment variables.")
