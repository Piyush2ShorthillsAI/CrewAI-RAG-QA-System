import json
import numpy as np
from pinecone_setup import index  # Pinecone index setup
from embedding import embedding_model  # Your embedding model


def query(user_query) :
# Load local storage (output.json)
  with open("output.json", "r", encoding="utf-8") as file:
       local_data = json.load(file)  # Load documents

# 🔍 Step 1: Query Pinecone
  query_embedding = np.array(embedding_model.encode(user_query)).tolist()  # Convert to list for Pinecone

  query_results = index.query(
       vector=query_embedding,
       top_k=5,  # Retrieve top 5 matches
       include_metadata=True,  # Ensure metadata is returned
       namespace="default")

# 🔍 Step 2: Extract Titles
  titles = []
  for match in query_results["matches"]:
     metadata = match.get("metadata", {})
     title = metadata.get("metadata_title") or metadata.get("title")  # Handle both cases
     if title:
        titles.append(title)

# 🔍 Step 3: Fetch Corresponding Content from output.json
  matching_data = []
  for title in titles:
      matched_doc = next((doc for doc in local_data if doc.get("title") == title), None)
      content = matched_doc["content"] if matched_doc else "No content found"

      matching_data.append({
          "title": title,
          "content": content })

# 🔍 Step 4: Print Extracted Data
  combined_content = "\n".join([doc["content"] for doc in matching_data])
  return combined_content