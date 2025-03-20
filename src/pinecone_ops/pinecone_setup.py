from pinecone import Pinecone
from pinecone_ops.config import Config
class PineconeManager:
    def __init__(self):
        self.access = Config()
        self.pinecone_client = Pinecone(api_key=self.access.PINECONE_API_KEY)
        self.index = self.get_index()

    def get_index(self):
        existing_indexes = self.pinecone_client.list_indexes()
        if self.access.INDEX_NAME not in [idx["name"] for idx in existing_indexes]:
            raise ValueError(f"Index '{self.access.INDEX_NAME}' not found. Create it manually in the Pinecone console.")
        return self.pinecone_client.Index(self.access.INDEX_NAME)

