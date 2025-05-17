import json
from unittest.mock import mock_open, patch
from llama_index.core import Document

# Assuming src is in PYTHONPATH or adjustments are made.
# For direct execution or simple test runners, this might be needed:
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.document_loader import DocumentLoader

# Use pytest fixtures instead of setUp
import pytest

# Define a fixture for the sample data
@pytest.fixture
def sample_data():
    return [
        {
            "doc_id_key": "doc1",
            "title": "First Title",
            "abstract": "First Abstract.",
            "year": "2023",
            "author": ["A. Writer", "B. Coder"],
            "url": "http://example.com/doc1"
        },
        {
            "doc_id_key": "doc2",
            "title": "Second Title",
            "year": 2024,
            "author": "Solo Author",
        },
        {
            "title": "Third Title",
            "abstract": "Third Abstract.",
            "year": "2025",
            "author": [],
            "url": "http://example.com/doc3"
        }
    ]

# Define a fixture for the DocumentLoader instance
@pytest.fixture
def document_loader(sample_data):
    corpus_path = "dummy_corpus.json"
    text_fields = ["title", "abstract"]
    metadata_fields = ["year", "author", "url"]
    id_field = "doc_id_key"
    return DocumentLoader(
        corpus_path=corpus_path,
        text_fields=text_fields,
        metadata_fields=metadata_fields,
        id_field=id_field
    )

# No need for a class inheriting from unittest.TestCase in pytest

@patch("builtins.open", new_callable=mock_open)
def test_load_data_success(mock_file_open, document_loader, sample_data):
    """Test successful loading and conversion of valid JSON data."""
    mock_file_open.return_value.read.return_value = json.dumps(sample_data)

    documents = document_loader.load_data()

    mock_file_open.assert_called_once_with(document_loader.corpus_path, 'r', encoding='utf-8')
    assert len(documents) == 3

    # Document 1
    doc1 = documents[0]
    assert isinstance(doc1, Document)
    assert doc1.doc_id == "doc1"
    expected_text1 = "title: First Title\nabstract: First Abstract.\n"
    assert doc1.text == expected_text1
    assert doc1.metadata == {"year": "2023", "author": ["A. Writer", "B. Coder"], "url": "http://example.com/doc1"}

    # Document 2 (missing abstract, int year, single author, missing URL)
    doc2 = documents[1]
    assert isinstance(doc2, Document)
    assert doc2.doc_id == "doc2"
    expected_text2 = "title: Second Title\nabstract: \n" # Ensure missing fields result in empty string for text part
    assert doc2.text == expected_text2
    # Note: Missing URL field should result in None in metadata
    assert doc2.metadata == {"year": 2024, "author": "Solo Author", "url": None}

    # Document 3 (auto-generated ID, empty author list)
    doc3 = documents[2]
    assert isinstance(doc3, Document)
    assert doc3.doc_id == "entry_2" # Auto-generated ID
    expected_text3 = "title: Third Title\nabstract: Third Abstract.\n"
    assert doc3.text == expected_text3
    assert doc3.metadata == {"year": "2025", "author": [], "url": "http://example.com/doc3"}

@patch("builtins.open", new_callable=mock_open)
@patch("builtins.print")
def test_load_data_file_not_found(mock_print, mock_file_open, document_loader):
    """Test behavior when the JSON file is not found."""
    mock_file_open.side_effect = FileNotFoundError

    documents = document_loader.load_data()

    assert len(documents) == 0
    mock_print.assert_any_call(f"Error: JSON file not found at {document_loader.corpus_path}. Please ensure it exists.")

@patch("builtins.open", new_callable=mock_open)
@patch("builtins.print")
def test_load_data_json_decode_error(mock_print, mock_file_open, document_loader):
    """Test behavior with malformed JSON content."""
    mock_file_open.return_value.read.return_value = "this is not json"

    documents = document_loader.load_data()

    assert len(documents) == 0
    assert any(f"Error decoding JSON from {document_loader.corpus_path}" in call_args[0][0] for call_args in mock_print.call_args_list)

@patch("builtins.open", new_callable=mock_open)
@patch("builtins.print")
def test_load_data_not_a_list(mock_print, mock_file_open, document_loader):
    """Test behavior when JSON content is not a list."""
    mock_file_open.return_value.read.return_value = json.dumps({"not": "a list"})

    documents = document_loader.load_data()

    assert len(documents) == 0
    mock_print.assert_any_call(f"Error: Expected a list of entries in {document_loader.corpus_path}, but got <class 'dict'>.")

@patch("builtins.open", new_callable=mock_open)
@patch("builtins.print")
def test_load_data_malformed_entry_in_list(mock_print, mock_file_open, document_loader, sample_data):
    """Test skipping a malformed entry (non-dict) within a list."""
    data_with_malformed_entry = [sample_data[0], "not a dict", sample_data[1]]
    mock_file_open.return_value.read.return_value = json.dumps(data_with_malformed_entry)

    documents = document_loader.load_data()

    assert len(documents) == 2 # Should load two valid documents
    assert documents[0].doc_id == "doc1"
    assert documents[1].doc_id == "doc2"
    mock_print.assert_any_call("Warning: Skipping entry #1 as it is not a dictionary: not a dict")

@patch("builtins.open", new_callable=mock_open)
@patch("builtins.print")
def test_load_data_empty_json_list(mock_print, mock_file_open, document_loader):
    """Test loading from an empty JSON list."""
    mock_file_open.return_value.read.return_value = json.dumps([])

    documents = document_loader.load_data()

    assert len(documents) == 0
    mock_print.assert_any_call(f"No documents were loaded from {document_loader.corpus_path}. Check the file content and format.") 