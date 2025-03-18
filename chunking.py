from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

def load_and_chunk_data(file_path="output.json", chunk_size=256, chunk_overlap=50):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunked_data = []
    for doc in data:
        chunks = text_splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            chunked_data.append({
                "id": f"{doc['url']}_chunk_{i}",
                "content": chunk,
                "metadata": {"url": doc["url"], "title": doc["title"]}
            })

    return chunked_data
