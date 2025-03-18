from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(chunked_data):
    """Generate embeddings for text content in chunked data."""
    embedded_data = []
    
    for idx, chunk in enumerate(chunked_data):
        if "content" in chunk:  # Ensure content key exists
            embedding_vector = embedding_model.encode(chunk["content"]).tolist()
            embedded_data.append({
                "id": str(idx),  # Convert index to string for Pinecone
                "values": embedding_vector,
                "metadata": {key: chunk[key] for key in chunk if key != "content"}  # Keep metadata
            })
        else:
            print(f"Warning: Skipping chunk {idx} (missing 'content' key)")

    return embedded_data
