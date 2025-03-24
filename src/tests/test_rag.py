import os
import sys

# Add src folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
import json

from data_processing.scrape_website import scrape
from data_processing.chunking import ChunkProcessor
from data_processing.embedding import EmbeddingGenerator
from llm_ops.llm2 import LLMHandler
from app import CrewAIApp
from unittest.mock import patch, MagicMock

@pytest.fixture
def output_file():
    """Provide a test output file path."""
    test_dir = "data/"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    return os.path.join(test_dir, "test_output.json")


# ------------------------ SCRAPER TESTS ------------------------

def test_scrape_output_file(output_file):
    """Test if output file is created after scrape."""
    start_url = "https://example.com"
    scrape(start_url, output_file)

    # Check if the output file is created
    assert os.path.exists(output_file)

    # Clean up
    if os.path.exists(output_file):
        os.remove(output_file)


def test_scrape_json_format(output_file):
    """Test if output file contains valid JSON format."""
    start_url = "https://example.com"
    scrape(start_url, output_file)

    # Check if output file is a valid JSON
    with open(output_file, "r") as f:
        data = json.load(f)
        assert isinstance(data, list)

    # Clean up
    if os.path.exists(output_file):
        os.remove(output_file)


def test_scraper():
    """Test scraping with valid URL."""
    start_url = "https://example.com"
    result = scrape(start_url)
    assert result is not None


def test_invalid_url():
    """Test invalid URL handling."""
    start_url = "invalid_url"
    with pytest.raises(ValueError):  # Handle ValueError or relevant exception
        scrape(start_url)


# ------------------------ CHUNK PROCESSOR TESTS ------------------------

def test_chunk_processor():
    """Test chunking of text."""
    chunk_processor = ChunkProcessor()
    text = "This is a sample text for testing chunking."
    chunks = chunk_processor.chunk_text(text, chunk_size=10)
    assert len(chunks) > 0
    assert isinstance(chunks, list)
    assert "This is a" in chunks[0]


def test_chunk_processor_empty_input():
    """Test chunk processor with empty text."""
    chunk_processor = ChunkProcessor()
    chunks = chunk_processor.chunk_text("", chunk_size=10)
    assert len(chunks) == 0


# ------------------------ EMBEDDING GENERATOR TESTS ------------------------

def test_embedding_processor():
    """Test embeddings for text."""
    embedding_generator = EmbeddingGenerator()
    text = "This is a sample text for embeddings."
    embeddings = embedding_generator.generate_embeddings([text])
    assert len(embeddings) > 0
    assert isinstance(embeddings[0], (list, tuple))


def test_embedding_empty_text():
    """Test embeddings with empty input."""
    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings([])
    assert len(embeddings) == 0


# ------------------------ LLM HANDLER TESTS ------------------------

def test_llm_response():
    """Test LLM handler response."""
    llm_handler = LLMHandler()
    user_query = "What is CrewAI?"
    context = "CrewAI is an AI platform that automates workflows."
    
    # Mock LLM response
    mock_response = {"response": "CrewAI automates workflows efficiently."}
    with patch.object(llm_handler, "query_llm", return_value=mock_response):
        response = llm_handler.query_llm(context, user_query)
        assert response["response"] == "CrewAI automates workflows efficiently."


def test_llm_empty_query():
    """Test LLM handler with empty query."""
    llm_handler = LLMHandler()
    user_query = ""
    context = "This is some context."

    with pytest.raises(ValueError):
        llm_handler.query_llm(context, user_query)


# ------------------------ CREWAI APP TESTS ------------------------

def test_crewai_app():
    """Mock CrewAIApp run."""
    with patch("app.CrewAIApp.run") as mock_run:
        app = CrewAIApp()
        app.run()
        mock_run.assert_called_once()


# ------------------------ EDGE CASES ------------------------

def test_chunk_overlap():
    """Test chunking with overlap."""
    chunk_processor = ChunkProcessor(chunk_size=20, chunk_overlap=5)
    text = "This is a longer sample text to test chunking with overlapping data."
    chunks = chunk_processor.chunk_text(text)
    assert len(chunks) > 1
    assert "This is a longer sample" in chunks[0]


def test_scrape_invalid_url():
    """Test handling of invalid URL."""
    start_url = "invalid_url"
    with pytest.raises(ValueError):
        scrape(start_url)


def test_chunk_file_creation(output_file):
    """Test if chunk file is created correctly."""
    chunk_processor = ChunkProcessor(file_path=output_file, file_name_chunks="data/test_chunks.json")
    chunk_processor.load_and_chunk_data()

    # Check if chunk file is created
    assert os.path.exists("data/test_chunks.json")

    # Clean up
    if os.path.exists("data/test_chunks.json"):
        os.remove("data/test_chunks.json")


def test_embedding_processor_empty_list():
    """Test embedding with an empty list."""
    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings([])
    assert len(embeddings) == 0
