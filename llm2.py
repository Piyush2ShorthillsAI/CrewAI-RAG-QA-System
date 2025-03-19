import requests
import json
from query_to_pinecone_final import query

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"


def load_llm():
    """
    Initializes the connection to Ollama API.
    Returns the API endpoint.
    """
    print("Ollama API connected successfully.")
    return OLLAMA_API_URL


def query_llm(context, user_query, llm_pipeline):
    """
    Queries the LLM with retrieved context and user query using Ollama API.
    """
    if not context.strip():
        return "I couldn't find relevant information in the retrieved documents."

    # Create the prompt
    prompt = (
        "You are an expert AI assistant answering queries based on the given context.\n"
        "Follow these rules:\n"
        "- Provide well-structured, fact-based answers.\n"
        "- If context lacks relevant data, state that explicitly.\n\n"
        f"Context:\n{context}\n\n"
        f"User Query:\n{user_query}\n\n"
        f"Answer:"
    )

    # Payload for Ollama API
    payload = {
        "model": "llama3",  # Use llama3 as the model name
        "prompt": prompt,
        "stream": False,  # Set to False to get a full response
        "options": {
            "temperature": 0.3,
            "max_tokens": 50
        }
    }

    # Send request to Ollama API
    try:
        response = requests.post(llm_pipeline, data=json.dumps(payload))
        response_data = response.json()

        if "response" in response_data:
            return response_data["response"].strip()
        else:
            return "Error: Invalid response from Ollama API."

    except Exception as e:
        return f"Error communicating with Ollama API: {str(e)}"