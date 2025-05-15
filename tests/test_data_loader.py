import os
import unittest
import json
from unittest.mock import mock_open, patch

# Ensure src directory is in Python path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_loader import AnthologyLoader
from llama_index.core import Document

class TestAnthologyLoader(unittest.TestCase):

    def setUp(self):
        self.dummy_json_path = "dummy_anthology_data.json"
        self.sample_data_valid = [
            {
                "ID": "W19-5001",
                "title": "A Great Paper Title",
                "author": ["Author One", "Author Two"],
                "year": "2023",
                "abstract": "This is the abstract of the first great paper.",
                "url": "http://example.com/paper1"
            },
            {
                "id": "P18-1001", # Note: using 'id' instead of 'ID' for this one
                "title": "Another Insightful Study",
                "author": "Solo Author", # Single author as string
                "year": 2022, # Year as int
                "abstract": "An abstract for the second study which is insightful.",
                "url": "http://example.com/paper2"
            },
            {
                "title": "Paper Without ID or Abstract",
                "author": [], # Empty author list
                "year": "2021"
                # Missing abstract, url, id
            }
        ]
        self.sample_data_malformed_entry = [
            {"title": "Good Paper"}, "This is not a dict entry"
        ]
        self.sample_data_not_a_list = {"key": "value"}

    def test_load_data_success(self):
        """Test successful loading and conversion of valid JSON data."""
        mock_json_content = json.dumps(self.sample_data_valid)
        with patch("builtins.open", mock_open(read_data=mock_json_content)) as mocked_file:
            loader = AnthologyLoader(json_file_path=self.dummy_json_path)
            documents = loader.load_data()
            
            mocked_file.assert_called_once_with(self.dummy_json_path, 'r', encoding='utf-8')
            self.assertEqual(len(documents), 3)
            
            # Check first document (W19-5001)
            doc1 = documents[0]
            self.assertIsInstance(doc1, Document)
            self.assertEqual(doc1.doc_id, "W19-5001")
            self.assertIn("Title: A Great Paper Title", doc1.text)
            self.assertIn("Abstract: This is the abstract of the first great paper.", doc1.text)
            self.assertEqual(doc1.metadata["title"], "A Great Paper Title")
            self.assertEqual(doc1.metadata["authors"], "Author One, Author Two")
            self.assertEqual(doc1.metadata["year"], "2023")
            self.assertEqual(doc1.metadata["url"], "http://example.com/paper1")
            self.assertEqual(doc1.metadata["source_file"], self.dummy_json_path)

            # Check second document (P18-1001)
            doc2 = documents[1]
            self.assertEqual(doc2.doc_id, "P18-1001")
            self.assertIn("Title: Another Insightful Study", doc2.text)
            self.assertEqual(doc2.metadata["authors"], "Solo Author")
            self.assertEqual(doc2.metadata["year"], "2022") # Should be converted to string

            # Check third document (Paper Without ID or Abstract)
            doc3 = documents[2]
            self.assertEqual(doc3.doc_id, "entry_2") # Auto-generated ID
            self.assertIn("Title: Paper Without ID or Abstract", doc3.text)
            self.assertIn("Abstract: No Abstract Provided", doc3.text)
            self.assertEqual(doc3.metadata["authors"], "Unknown Authors")
            self.assertEqual(doc3.metadata["url"], "No URL Provided")

    def test_load_data_file_not_found(self):
        """Test behavior when the JSON file is not found."""
        with patch("builtins.open", mock_open()) as mocked_file:
            mocked_file.side_effect = FileNotFoundError
            with patch('builtins.print') as mock_print: # Suppress print
                loader = AnthologyLoader(json_file_path="non_existent_file.json")
                documents = loader.load_data()
                self.assertEqual(len(documents), 0)
                mock_print.assert_any_call("Error: JSON file not found at non_existent_file.json. Please ensure it exists.")

    def test_load_data_json_decode_error(self):
        """Test behavior with malformed JSON content."""
        malformed_json_content = "{\"title\": \"Unfinished Paper\", " # Intentionally malformed
        with patch("builtins.open", mock_open(read_data=malformed_json_content)) as mocked_file:
            with patch('builtins.print') as mock_print: # Suppress print
                loader = AnthologyLoader(json_file_path=self.dummy_json_path)
                documents = loader.load_data()
                self.assertEqual(len(documents), 0)
                # mock_print.assert_any_call(f"Error decoding JSON from {self.dummy_json_path}: Expecting property name enclosed in double quotes: line 1 column 28 (char 27)") # Message can vary slightly
                # More robust check for the print call:
                called_once_with_error = False
                for call_args in mock_print.call_args_list:
                    args, _ = call_args
                    if args and isinstance(args[0], str) and \
                       f"Error decoding JSON from {self.dummy_json_path}" in args[0] and \
                       "Expecting property name enclosed in double quotes" in args[0]: # Or a more generic part of json error
                        called_once_with_error = True
                        break
                self.assertTrue(called_once_with_error, f"Expected print call with JSON decode error message for {self.dummy_json_path} not found.")

    def test_load_data_not_a_list(self):
        """Test behavior when JSON content is not a list as expected."""
        mock_json_content = json.dumps(self.sample_data_not_a_list)
        with patch("builtins.open", mock_open(read_data=mock_json_content)) as mocked_file:
            with patch('builtins.print') as mock_print: # Suppress print
                loader = AnthologyLoader(json_file_path=self.dummy_json_path)
                documents = loader.load_data()
                self.assertEqual(len(documents), 0)
                mock_print.assert_any_call(f"Error: Expected a list of entries in {self.dummy_json_path}, but got <class 'dict'>.")

    def test_load_data_malformed_entry_in_list(self):
        """Test skipping a malformed entry within a list of entries."""
        mock_json_content = json.dumps(self.sample_data_malformed_entry)
        with patch("builtins.open", mock_open(read_data=mock_json_content)) as mocked_file:
            with patch('builtins.print') as mock_print: # Suppress print
                loader = AnthologyLoader(json_file_path=self.dummy_json_path)
                documents = loader.load_data()
                self.assertEqual(len(documents), 1) # Only the valid dict entry should be processed
                self.assertEqual(documents[0].metadata["title"], "Good Paper")
                mock_print.assert_any_call("Warning: Skipping entry #1 as it is not a dictionary: This is not a dict entry")

    def test_load_data_empty_json_list(self):
        """Test loading from an empty JSON list."""
        mock_json_content = json.dumps([])
        with patch("builtins.open", mock_open(read_data=mock_json_content)) as mocked_file:
            with patch('builtins.print') as mock_print: # Suppress print
                loader = AnthologyLoader(json_file_path=self.dummy_json_path)
                documents = loader.load_data()
                self.assertEqual(len(documents), 0)
                mock_print.assert_any_call(f"No documents were loaded from {self.dummy_json_path}. Check the file content and format.")

if __name__ == "__main__":
    unittest.main() 