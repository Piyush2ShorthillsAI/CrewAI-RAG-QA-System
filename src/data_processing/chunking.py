from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

class ChunkProcessor:
    def __init__(self, file_path="data/output.json", chunk_size=256, chunk_overlap=50):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def load_and_chunk_data(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
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