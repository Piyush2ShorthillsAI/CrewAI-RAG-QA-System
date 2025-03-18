import streamlit as st
from llm2 import load_llm
from query_to_pinecone_final import query
from llm2 import query_llm
import os
from log import log_interaction
import pandas as pd

st.title("💬 CrewAI Q&A System")
st.write("Ask a question, and the LLM will generate a response based on the retrieved context.")
user_query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if user_query:
        llm_pipeline = load_llm()

        # Query Pinecone for relevant chunks
        context = query(user_query)
        
        # Get LLM response
        response = query_llm(context, user_query, llm_pipeline)
        
        st.write("Answer:")
        st.success(response)

        # Log interaction
        log_interaction(query,response)
    else:
        st.warning("Please enter a question.")

if os.path.isfile("logs.csv"):
    st.subheader("📜 Previous Queries")
    try:
        df = pd.read_csv("logs.csv")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error reading log file: {e}")
else:
    st.warning("No logs found.")