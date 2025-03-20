from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, chunked_data):
        embedded_data = []
        
        for idx, chunk in enumerate(chunked_data):
            if "content" in chunk:
                embedding_vector = self.embedding_model.encode(chunk["content"]).tolist()
                embedded_data.append({
                    "id": str(idx),
                    "values": embedding_vector,
                    "metadata": {key: chunk[key] for key in chunk if key != "content"}
                })
        
        return embedded_data
