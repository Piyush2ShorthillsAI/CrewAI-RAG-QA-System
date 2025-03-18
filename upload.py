from pinecone_setup import index
import numpy as np
from embedding import embedding_model

def flatten_metadata(metadata):
    """Flattens nested metadata dictionaries into a valid Pinecone format."""
    flat_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_metadata[f"{key}_{sub_key}"] = str(sub_value)
        elif isinstance(value, list):
            flat_metadata[key] = " ".join(map(str, value))  # ✅ Ensure list is stored as a string
        else:
            flat_metadata[key] = str(value) if isinstance(value, (dict, list)) else value# ✅ Convert everything else to a string
    return flat_metadata

def upload_to_pinecone(embeddings, batch_size=100, namespace="default"):
    """
    Uploads embedding vectors to Pinecone in batches.

    :param embeddings: List of dictionaries containing "id", "values", and "metadata".
    :param batch_size: Number of vectors per batch.
    :param namespace: Pinecone namespace.
    """
    if not embeddings:
        print("❌ No embeddings to upload.")
        return

    # Load the JSON file
    import json
    with open("output.json", "r", encoding="utf-8") as file:
        documents = json.load(file)  # Ensure it's a list of dictionaries

    formatted_embeddings = [
        {
            "id": str(i),  # Unique ID for each document
            "values": np.array(embedding_model.encode(doc["content"])).tolist(),  # Ensure embedding is a NumPy array
            "metadata": flatten_metadata({
                "title": doc.get("title", "Unknown Title"),
                "url": doc.get("url", "No URL"),
                "text": doc.get("content", "")[:39000]# Ensure content exists
            })
        }
        for i, doc in enumerate(documents)
    ]

    # Debugging: Check sample before upload
    if formatted_embeddings:
        print("🔍 Sample Vector:", formatted_embeddings[0])

    # Upload in batches
    for i in range(0, len(formatted_embeddings), batch_size):
        batch = formatted_embeddings[i : i + batch_size]
        try:
            index.upsert(vectors=batch, namespace=namespace)
            print(f"Batch {i // batch_size + 1} uploaded successfully.")
        except Exception as e:
            print(f"❌ Error uploading batch {i // batch_size + 1}: {e}")

    print("✅ Upload to Pinecone completed.")
    print(f"Total vectors in index: {index.describe_index_stats()}")