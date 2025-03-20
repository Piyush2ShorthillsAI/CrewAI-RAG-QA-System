import streamlit as st
import os
import pandas as pd
from llm_ops.llm2 import LLMHandler
from llm_ops.query_to_pinecone_final import PineconeQueryHandler
from logger.log import InteractionLogger

# Define the CSS style for the main content area only
background_style = """
<style>
    /* Set AI-themed background for the main content area */
    .main {
        background-color: #1e1e2f; /* Deep AI blue */
        color: #ffffff; /* White text for better contrast */
        padding: 10px;
    }

    /* Ensure sidebar remains unaffected */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa; /* Default sidebar color */
    }
</style>
"""


class CrewAIApp:
    def __init__(self):
        self.llm_handler = LLMHandler()
        self.pinecone_handler = PineconeQueryHandler()
        self.logger = InteractionLogger()
        self.log_file = "src/logger/logs.csv"
    def display_sidebar(self):
        
        st.sidebar.title("What You Can Ask❓")
        st.sidebar.write(
            """
            ✅ **Explore Topics:**
            - 🔍 **Web Scraping:** "What data is extracted from CrewAI?"
            - 📄 **Document Insights:** "Summarize the content of a page."
            - 🧠 **RAG Process:** "How does the embedding and querying work?"
            - 🔗 **Pinecone Information:** "What embeddings are stored in Pinecone?"

            ⚡️ **Example Queries:**
            - "Explain the CrewAI architecture."
            - "Summarize the content of the scraped webpage."
            - "List top 3 key points from the data."

            ✨ **Pro Tips:**
            - Keep your queries specific for better accuracy.
            - Ask for summaries if you need concise answers.
            """
        )
       

    def display_title(self):
        """Displays the application title and introduction."""
        st.markdown(background_style, unsafe_allow_html=True)
        st.markdown("<div class='main'>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align:left; color: #4CAF50;'>✨ CrewAI Q&A System</h1>", unsafe_allow_html=True)
        st.write("Welcome to the AI-powered Q&A system. Ask your questions below.")
        


    def get_user_input(self):
        """Captures user input from the text box."""
        return st.text_input("Enter your query:")

    def process_query(self, user_query):
        """Processes user query by querying Pinecone and LLM."""
        if user_query:
            # Query Pinecone for relevant context
            context = self.pinecone_handler.query(user_query)

            # Get LLM response
            response = self.llm_handler.query_llm(context, user_query)

            # Display the response
            st.write("Answer:")
            st.success(response)

            # Log the interaction
            self.logger.log_interaction(user_query, response)
        else:
            st.warning("Please enter a question.")

    def display_logs(self):
        """Displays previous query logs if available."""
        if os.path.isfile(self.log_file):
            st.subheader("📜 Previous Queries")
            try:
                df = pd.read_csv(self.log_file)
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error reading log file: {e}")
        else:
            st.warning("No logs found.")

    def run(self):
        """Runs the Streamlit application."""
        self.display_sidebar() 
        self.display_title()
        user_query = self.get_user_input()

        if st.button("Get Answer"):
            self.process_query(user_query)

        self.display_logs()


# Run the application
if __name__ == "__main__":
    app = CrewAIApp()
    app.run()
