import pytest
import json
import os
import tempfile
from unittest.mock import patch
from data_processing.scrape_website import scrape
from data_processing.chunking import ChunkProcessor
from data_processing.embedding import EmbeddingGenerator
from llm_ops.llm2 import LLMHandler
from app import CrewAIApp
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TestScraper:
    def test_scrape_json_format(self):
        """Test if the output file contains valid JSON format using a manually created JSON."""
        test_data = [
            {"url": "https://example.com", "title": "Example", "content": "Test content"}
        ]
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_file:
            output_file = temp_file.name
            json.dump(test_data, temp_file)
            temp_file.close()
        
        assert os.path.exists(output_file)

        with open(output_file, "r") as f:
            data = json.load(f)
        
        assert isinstance(data, list)
        assert len(data) > 0
        assert all(isinstance(item, dict) for item in data)
        
        os.remove(output_file)

    def test_scraper(self):
        url = "https://docs.crewai.com"
        output_file = "data/test_output.json"
        
        scrape(url, output_file)
        
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0

        with open(output_file, "r") as f:
            data = json.load(f)
        
        assert isinstance(data, list)
        assert len(data) > 0
        assert all(isinstance(item, dict) for item in data)
        
        os.remove(output_file)


class TestChunkProcessor:
    def test_chunk_processor(self):
        chunk_processor = ChunkProcessor(
            file_path="data/output.json",
            file_name_chunks="data/chunks.json",
            chunk_size=256,
            chunk_overlap=50
        )
        chunked_data = chunk_processor.load_and_chunk_data()
        assert len(chunked_data) > 0

    def test_chunk_processor_empty_input(self, tmp_path):
        empty_file_path = tmp_path / "empty_data.json"
        with open(empty_file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        chunk_processor = ChunkProcessor(
            file_path=str(empty_file_path),
            file_name_chunks="data/chunks.json",
            chunk_size=256,
            chunk_overlap=50
        )
        chunks = chunk_processor.load_and_chunk_data()
        assert len(chunks) == 0


class TestEmbeddingGenerator:
    def test_embedding_processor(self):
        embedding_generator = EmbeddingGenerator()
        text = {
            "url": "https://docs.crewai.com/tools/ragtool",
            "title": "RAG Tool - CrewAI",
            "content": "The tool is a dynamic knowledge base tool for answering questions using Retrieval-Augmented Generation."
        }
        embeddings = embedding_generator.generate_embeddings([text])
        assert len(embeddings) > 0
        assert isinstance(embeddings[0], dict)
        assert "id" in embeddings[0]
        assert "values" in embeddings[0]
        assert "metadata" in embeddings[0]
        assert isinstance(embeddings[0]["values"], (list, tuple))
        assert len(embeddings[0]["values"]) > 0
        assert isinstance(embeddings[0]["metadata"], dict)

    def test_embedding_empty_text(self):
        embedding_generator = EmbeddingGenerator()
        embeddings = embedding_generator.generate_embeddings([])
        assert len(embeddings) == 0


class TestLLMHandler:
    def test_llm_response(self):
        llm_handler = LLMHandler()
        user_query = "What is CrewAI?"
        context = "CrewAI is an AI platform that automates workflows."
        mock_response = {"response": "CrewAI automates workflows efficiently."}
        with patch.object(llm_handler, "query_llm", return_value=mock_response):
            response = llm_handler.query_llm(context, user_query)
            assert response['response'] == "CrewAI automates workflows efficiently."

    def test_llm_empty_query(self):
        llm_handler = LLMHandler()
        user_query = ""
        context = "This is some context."
        with pytest.raises(ValueError):
            llm_handler.query_llm(context, user_query)


class TestCrewAIApp:
    def test_crewai_app(self):
        with patch("app.CrewAIApp.run") as mock_run:
            app = CrewAIApp()
            app.run()
            mock_run.assert_called_once()


class TestEdgeCases:
    def test_chunk_overlap(self):
        chunk_processor = ChunkProcessor(
            file_path="data/output.json",
            file_name_chunks="data/chunks.json",
            chunk_size=40,
            chunk_overlap=5
        )
        text = "This is a longer sample text to test chunking with overlapping data."
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_processor.chunk_size,
            chunk_overlap=chunk_processor.chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        assert len(chunks) > 1
        expected_substring = "This is a longer sample"
        found = any(expected_substring in chunk for chunk in chunks)
        assert found

    def test_chunk_file_creation(self):
        output_data = [
            {
                "url": "https://docs.crewai.com/tools/seleniumscrapingtool",
                "title": "Selenium Scraper - CrewAI",
                "content": "The tool is designed to extract and read the content of a specified website using Selenium."
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_file:
            output_file = temp_file.name
            json.dump(output_data, temp_file, indent=4)
        chunk_processor = ChunkProcessor(file_path=output_file, file_name_chunks="data/test.json")
        chunk_processor.load_and_chunk_data()
        try:
            assert os.path.exists("data/test.json")
            with open("data/test.json", "r") as f:
                chunks_data = json.load(f)
            assert isinstance(chunks_data, (list, dict))
            assert len(chunks_data) > 0
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)
            if os.path.exists("data/test.json"):
                os.remove("data/test.json")

    def test_embedding_processor_empty_list(self):
        embedding_generator = EmbeddingGenerator()
        embeddings = embedding_generator.generate_embeddings([])
        assert len(embeddings) == 0
