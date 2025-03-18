from scrape_website import scrape
from chunking import load_and_chunk_data
from embedding import generate_embeddings
from upload import upload_to_pinecone

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

def main():
    # Step 1: Run the web crawler to scrape the website
    url = 'https://docs.crewai.com'

    scrape(start_url=url)
    print(f"Scraped website: {url}")

    chunked_data = load_and_chunk_data()
    print("chunking is done..")
     
    chunked_data = generate_embeddings(chunked_data)
    print("Embeddings are generated..")

    upload_to_pinecone(chunked_data)
    print("All chunks uploaded successfully.")
   
if __name__ == "__main__":
    main()
