# CrewAI Q&A System (RAG-based)

This project implements a **Retrieval-Augmented Generation (RAG)** system using **Mistral-7B** for answering user queries based on retrieved context. The system fetches relevant text chunks from **Pinecone**, processes them with an **LLM (Mistral-7B)**, and provides structured answers. The interface is built using **Streamlit** for easy interaction.

## Features

- **Web-based Q&A system** using Streamlit
- **Retrieval-Augmented Generation (RAG)** using Pinecone
- **Mistral-7B LLM** for text generation
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
pip install torch transformers pinecone-client streamlit pandas
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/crewai-qa-system.git
cd crewai-qa-system
```

### 2. Set Up Pinecone

- Create a **Pinecone** account and get your API key.
- Configure your index in `query_to_pinecone_final.py`.

### 2.1 Set Up .env File

- Create a `.env` file in the root directory.
- Add the following line with your Pinecone API key:

```
PINECONE_API_KEY = ""
```

- Create a **Pinecone** account and get your API key.
- Configure your index in `query_to_pinecone_final.py`.

### 3. Run the Application

The system uses **Mistral-7B** as the RAG model. The model name used is:

```
model_name = "mistralai/Mistral-7B-v0.1"
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
python main_final.py

# Step 3: Launch the Streamlit UI
streamlit run streamlit.py
```

## File Structure

```
websitescraper/
├─ .env
├─ chunking.py
├─ config.py
├─ embedding.py
├─ generated_test_cases.json
├─ llm2.py
├─ log.py
├─ logs.csv
├─ main_final.py  # Scrapes website and generates `output.json`
├─ output.json    # Generated scraped data
├─ pinecone_setup.py
├─ query_to_pinecone_final.py
├─ scrape_website.py
├─ streamlit.py   # Runs the Streamlit UI
├─ test_cases.py
├─ testing.py
├─ upload.py
└─ venv
```

## Usage

1. Run `main_final.py` to **scrape the website** and generate `output.json`, then after this the data undergoes chunking and embedding before being stored in Pinecone.
2. Open the **Streamlit UI**.

To run the `streamlit.py` file and open the Streamlit UI, use the following command:

```bash
streamlit run streamlit.py
```

This launches the UI where users can enter queries and receive responses.

3. Enter a question in the text input field.
4. The system retrieves relevant information and generates an answer.
5. Previous queries are displayed in a table.

## Logging

User queries and responses are saved in `logs.csv`. You can view them directly in the UI.

## Optimizations for CPU

- **Disabled gradients for inference**
- **Used ********`torch.set_num_threads()`******** for better CPU performance**
- **Reduced ********`max_new_tokens`******** to speed up response generation**

📊 Evaluation Results: 60% Accuracy | Passed: 30 | Failed: 20

## Contributing

Feel free to open an issue or submit a pull request for improvements!



