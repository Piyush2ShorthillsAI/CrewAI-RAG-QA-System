import requests
import json
from pinecone_ops.config import Config

class LLMHandler:
    def __init__(self):
        self.access = Config()
        self.api_url = self.access.OLLAMA_API_URL
    

    def load_llm(self):
       """
       Initializes the connection to Ollama API.
       Returns the API endpoint.
       """
       print("Ollama API connected successfully.")
       return self.api_url



    def query_llm(self, context, user_query, model_name="llama3", temperature=0.3, max_tokens=50):
        """
        Queries the LLM with retrieved context and user query using Ollama API.

        Args:
            context (str): Context retrieved from Pinecone.
            user_query (str): User query to be answered by the LLM.
            model_name (str): Model to use (default: llama3).
            temperature (float): Controls randomness in output (0.0 - 1.0).
            max_tokens (int): Maximum token length for the response.

        Returns:
            str: LLM-generated response or error message.
        """
        if not context.strip():
            return "No relevant information found in the retrieved documents."

        # Create the prompt
        prompt = (
            "You are an expert AI assistant answering queries based on the given context.\n"
            "Follow these rules:\n"
            "- Provide well-structured, fact-based answers.\n"
            "- If context lacks relevant data, state that explicitly.\n\n"
            f"Context:\n{context}\n\n"
            f"User Query:\n{user_query}\n\n"
            "Answer:"
        )

        # Payload for Ollama API
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,  # Set to False to get a full response
            "options": {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }

        # Send request to Ollama API
        try:
            response = requests.post(self.api_url, data=json.dumps(payload))
            response_data = response.json()

            if "response" in response_data:
                return response_data["response"].strip()
            else:
                return "Error: Invalid response from Ollama API."

        except Exception as e:
            return f"Error communicating with Ollama API: {str(e)}"

