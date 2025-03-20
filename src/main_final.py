from data_processing.scrape_website import scrape
from data_processing.chunking import ChunkProcessor
from data_processing.embedding import EmbeddingGenerator
from pinecone_ops.pinecone_setup import PineconeManager
from pinecone_ops.upload import PineconeUploader

class RAGPipeline:
    def __init__(self, url):
        self.url = url
        self.scraper = scrape
        self.chunk_processor = ChunkProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.pinecone_manager = PineconeManager()
        self.uploader = PineconeUploader()
    
    def run_pipeline(self):
        print(f"Starting scraping for URL: {self.url}")
        self.scraper(self.url)
        
        print("Chunking content...")
        chunked_data = self.chunk_processor.load_and_chunk_data()
        
        print("Generating embeddings...")
        embedded_data = self.embedding_generator.generate_embeddings(chunked_data)
        
        print("Uploading embeddings to Pinecone...")
        self.uploader.upload_to_pinecone(embedded_data)
        
        print("RAG pipeline completed.")
        
if __name__ == "__main__":
    url = 'https://docs.crewai.com'
    pipeline = RAGPipeline(url)
    pipeline.run_pipeline()
