# --- query_to_pinecone_final.py ---
import json
import numpy as np
from pinecone_ops.pinecone_setup import PineconeManager
from data_processing.embedding import EmbeddingGenerator


class PineconeQueryHandler:
    def __init__(self, index_namespace="default"):
        self.pinecone_manager = PineconeManager()
        self.index = self.pinecone_manager.index
        self.embedding_generator = EmbeddingGenerator()
        self.index_namespace = index_namespace

    def query(self, user_query, top_k=5):
        """
        Queries Pinecone with the user query and retrieves relevant documents.

        Args:
            user_query (str): The query entered by the user.
            top_k (int): Number of top matches to retrieve.

        Returns:
            str: Combined content from retrieved documents.
        """
        # Load local storage (output.json)

        with open("/home/shtlp_0170/Pictures/Rag-Project/data/output.json", "r", encoding="utf-8") as file:
            local_data = json.load(file)

        # 🔍 Step 1: Generate query embedding
        query_embedding = np.array(self.embedding_generator.embedding_model.encode(user_query)).tolist()

        # 🔍 Step 2: Query Pinecone
        query_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self.index_namespace
        )

        # 🔍 Step 3: Extract titles from matched results
        titles = []
        for match in query_results["matches"]:
            metadata = match.get("metadata", {})
            title = metadata.get("metadata_title") or metadata.get("title")
            if title:
                titles.append(title)

        # 🔍 Step 4: Fetch corresponding content from output.json
        matching_data = []
        for title in titles:
            matched_doc = next((doc for doc in local_data if doc.get("title") == title), None)
            content = matched_doc["content"] if matched_doc else "No content found"
            matching_data.append({
                "title": title,
                "content": content
            })

        # 🔍 Step 5: Combine content from matching documents
        combined_content = "\n".join([doc["content"] for doc in matching_data])
        return combined_content
