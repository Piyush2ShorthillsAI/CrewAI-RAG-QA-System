from pinecone_ops.pinecone_setup import PineconeManager
import numpy as np

class PineconeUploader:
    def __init__(self):
        self.pinecone_manager = PineconeManager()
        self.index = self.pinecone_manager.index
       
        
    def flatten_metadata(self, metadata):
        flat_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_metadata[f"{key}_{sub_key}"] = str(sub_value)
            elif isinstance(value, list):
                flat_metadata[key] = " ".join(map(str, value))
            else:
                flat_metadata[key] = str(value)
        return flat_metadata

    def upload_to_pinecone(self, embeddings, batch_size=100, namespace="default"):
        if not embeddings:
            print("No embeddings to upload.")
            return
        
        formatted_embeddings = [
            {
                "id": str(i),
                "values": np.array(embedding["values"]).tolist(),
                "metadata": self.flatten_metadata(embedding["metadata"])
            }
            for i, embedding in enumerate(embeddings)
        ]
        
        for i in range(0, len(formatted_embeddings), batch_size):
            batch = formatted_embeddings[i: i + batch_size]
            try:
                self.index.upsert(vectors=batch, namespace=namespace)
                print(f"Batch {i // batch_size + 1} uploaded successfully.")
            except Exception as e:
                print(f"Error uploading batch {i // batch_size + 1}: {e}")
        
        print(f"Upload to Pinecone completed. Total vectors: {self.index.describe_index_stats()}")
