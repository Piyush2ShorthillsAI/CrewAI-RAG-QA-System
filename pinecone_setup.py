from pinecone import Pinecone
from config import PINECONE_API_KEY,INDEX_NAME

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists
existing_indexes = pinecone_client.list_indexes()

if INDEX_NAME not in [idx["name"] for idx in existing_indexes]:
    print(f"Index '{INDEX_NAME}' not found. Please create it manually in the Pinecone console.")

# Connect to the existing index
index = pinecone_client.Index(INDEX_NAME)  