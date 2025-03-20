# --- config.py ---
import os
from dotenv import load_dotenv


class Config:
    """
    Config class to manage environment variables and API settings.
    """

    def __init__(self):
        load_dotenv()
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENV = "us-east-1"
        self.INDEX_NAME = "end-to-end-rag"
        self.OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")

    def get_pinecone_config(self):
        """
        Returns Pinecone configuration parameters.
        """
        return {
            "api_key": self.PINECONE_API_KEY,
            "env": self.PINECONE_ENV,
            "index_name": self.INDEX_NAME,
        }

    def get_ollama_url(self):
        """
        Returns Ollama API URL.
        """
        return self.OLLAMA_API_URL


# # Instantiate the config object
# config = Config()
