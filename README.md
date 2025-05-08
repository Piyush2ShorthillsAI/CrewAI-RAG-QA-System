# CrewAI Q&A System (RAG-based)

This project implements a **Retrieval-Augmented Generation (RAG)** system using **llama3** for answering user queries based on retrieved context. The system fetches relevant text chunks from **Pinecone**, processes them with an **LLM (llama3)**, and provides structured answers. The interface is built using **Streamlit** for easy interaction.

## Features

- **Web-based Q&A system** using Streamlit
- **Retrieval-Augmented Generation (RAG)** using Pinecone
- **llama** for text generation
- **Efficient CPU-based inference**
- **Logging** of user interactions

## Architecture Overview

The architecture consists of the following components:

- **Web Scraping Module:** Extracts textual data from websites using Selenium and BeautifulSoup.
- **Chunking Module:** Splits extracted text into chunks using RecursiveCharacterTextSplitter.
- **Embedding Generation Module:** Uses SentenceTransformers to generate embeddings for text chunks.
- **Vector Database:** Stores embeddings using Pinecone for retrieval.
- **Query Processing Module:**
    - Embeds user queries.
    - Matches query embeddings with stored embeddings in Pinecone.
    - Generates context for the LLM.
- **LLM-based Answer Generation:** Uses a transformer model to generate responses based on retrieved context.

### [View Architecture](https://drive.google.com/file/d/1C4y46qnUudbCHrfRpXOzga61xkrnRm1V/view?usp=sharing)

## Installation

### Prerequisites

Ensure you have **Python 3.8+** installed. Also, install the necessary dependencies:

```bash
pip install requirement.txt 
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Piyush2ShorthillsAI/CrewAI-RAG-QA-System.git
cd crewai-qa-system
```

### 2. Set Up Pinecone

- Create a **Pinecone** account and get your API key.
- Configure your index in `src/pinecone_ops/pinecone_setup.py`.

### 2.1 Set Up .env File

- Create a `.env` file in the root directory.
- Add the following line with your Pinecone API key:

```
PINECONE_API_KEY = ""
```

### 3. Run the Application

The system uses **llama3 ** as the RAG model. The model name used is:

```
model_name = "llama3"
```

The embedding model used is **all-MiniLM-L6-v2** from Sentence Transformers. It is loaded as:

```python
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
```

The chunking process is handled using **LangChain's RecursiveCharacterTextSplitter**. It is used because it effectively breaks down large texts into manageable chunks while maintaining semantic context. This approach ensures that relevant information is preserved across splits, improving retrieval accuracy during query processing. It is loaded as:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define text splitter
chunk_size = 256  # Optimal size to ensure relevant context is captured without exceeding token limits
chunk_overlap = 50  # Ensures overlapping content to maintain context between chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
```

- **chunk\_size**: Set to 256, ensuring the chunks are small enough to fit into the model's context window while maintaining relevant information.
- **chunk\_overlap**: Set to 50, providing sufficient overlap to prevent loss of context across adjacent chunks.

The system requires scraping first, then querying the processed data via Streamlit.

```bash
# Step 1: Scrape the website and generate `output.json`
python3 src/main_final.py

# Step 3: Launch the Streamlit UI
streamlit run src/app.py
```

## File Structure

```
📂 CrewAI-RAG-QA-System/
├── 📚 venv/                           # Virtual environment (isolated Python packages)
├── 📂 src/                            # Source code
│   ├── 📂 data_processing/            # Data preparation and preprocessing
│   │   ├── 📄 chunking.py               # Text chunking logic for splitting data
│   │   ├── 📄 embedding.py              # Generate embeddings using Hugging Face models
│   │   ├── 📄 scrape_website.py         # Web scraping logic with BeautifulSoup/Selenium
│   ├── 📂 pinecone_ops/               # Pinecone-related operations
│   │   ├── 📄 pinecone_setup.py         # Pinecone initialization and setup
│   │   ├── 📄 config.py                 # Configuration for Pinecone and storage
│   │   ├── 📄 upload.py                 # Upload data to Pinecone/Cloud Storage
│   ├── 📂 tests/                      # Unit and integration tests
│   │   ├── 📄 test_rag.py               # Pytest test cases for the application
│   ├── 📂 logger/                     # Logging and monitoring
│   │   ├── 📄 log.py                    # Logging utilities and handlers
│   │   ├── 📄 logs.csv                  # Log storage for audit trails
│   ├── 📂 llm_ops/                    # LLM interaction and query logic
│   │   ├── 📄 llm2.py                   # LLM pipeline with RAG logic
│   │   ├── 📄 query_to_pinecone.py      # Querying Pinecone and retrieving context
│   ├── 📄 app.py                      # Main Streamlit/Flask entry point
│   ├── 📄 main_final.py               # Main script for execution and deployment
│   ├── 📄 testing.py                  # Test script for evaluating model on input file
│   ├── 📂 test_results/               # Test evaluation results
│   │   ├── 📄 bert_base_scores_.xlsx   # Query-wise and final RAG evaluation scores
├── 📂 data/                           # Data storage and input/output files
│   ├── 📄 output.json                  # Processed JSON output with extracted data
│   ├── 📄 q&a_rag_application.xlsx     # Input XLSX file with test cases
│   ├── 📄 chunks.json                  # Chunked data ready for embedding
├── 📂 env/                           # Environment variables for secure configuration
├── 📄 .gitignore                     # Git settings to ignore unnecessary files
├── 📄 README.md                      # Project documentation and usage guide
└── 📄 requirements.txt               # List of required dependencies


## Usage

1. Run `main_final.py` to **scrape the website** and generate `output.json`, then after this the data undergoes chunking and embedding before being stored in Pinecone.
2. Open the **Streamlit UI**.

To run the `app.py` file and open the Streamlit UI, use the following command:

```bash
streamlit run src/app.py
```

This launches the UI where users can enter queries and receive responses.

3. Enter a question in the text input field.
4. The system retrieves relevant information and generates an answer.
5. Previous queries are displayed in a table.

## Logging

User queries and responses are saved in `logger/logs.csv`. You can view them directly in the UI.

## Evaluation Results
- **Final Evaluation Summary**
   - **Total Score: 0.56**
   - **Average Final Query-Wise Score: 0.52**
   - **Average Answer Correctness Score (Considered Final in Some Cases): 0.59**

## Contributing

Feel free to open an issue or submit a pull request for improvements!

