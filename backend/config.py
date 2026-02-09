import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration settings for the RAG system
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Document processing settings
CHUNK_SIZE = 800       # Size of text chunks for vector storage
CHUNK_OVERLAP = 100     # Characters to overlap between chunks
MAX_RESULTS = 5         # Maximum search results to return
MAX_HISTORY = 2         # Number of conversation messages to remember

# Database paths
CHROMA_PATH = "./chroma_db"  # ChromaDB storage location
