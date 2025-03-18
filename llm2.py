import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from query_to_pinecone_final import query

# Optimize CPU performance
torch.set_num_threads(8)  # Adjust based on your CPU core count
torch.set_grad_enabled(False)  # Disable gradients for inference


def load_llm():
    model_name = "mistralai/Mistral-7B-v0.1"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cpu"  # Removed 8-bit quantization
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cpu"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def query_llm(context, user_query, llm_pipeline):
    """
    Queries the LLM with retrieved context and user query.
    """
    if not context.strip():
        return "I couldn't find relevant information in the retrieved documents."
    
    prompt = (
        "You are an expert AI assistant answering queries based on the given context.\n"
        "Follow these rules:\n"
        "- Provide well-structured, fact-based answers.\n"
        "- If context lacks relevant data, state that explicitly.\n\n"
        f"Context:\n{context}\n\n"
        f"User Query:\n{user_query}\n\n"
        f"Answer:"
    )
    response = llm_pipeline(prompt,max_new_tokens = 50,do_sample=True, temperature=0.3, pad_token_id=0)
    
    return response[0]["generated_text"]


def query_llm_with_pinecone(user_query,llm_pipeline):
    """
    Retrieves relevant context from Pinecone and queries the LLM.
    """
    retrieved_results = query(user_query)
    
    print(f"[DEBUG] Final retrieved_results: {retrieved_results}")

    return query_llm(retrieved_results, user_query, llm_pipeline)
