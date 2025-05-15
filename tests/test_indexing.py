import os
import unittest
import shutil
import tempfile
from unittest.mock import patch, MagicMock

# Ensure src and scripts directories are in Python path for imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../scripts')))

from llama_index.core import Document, VectorStoreIndex, Settings
# from llama_index.core.base.embeddings import BaseEmbedding # Old attempt
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # Import concrete class for checking

from core_components import initialize_embedding_model, DEFAULT_EMBED_MODEL_NAME
from build_index import build_and_persist_index # build_index script
# We need AnthologyLoader to mock its load_data method
from data_loader import AnthologyLoader 

class TestIndexingPipeline(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory for storing test index."""
        self.test_storage_dir = tempfile.mkdtemp(prefix="test_idx_storage_")
        self.dummy_json_path = "dummy_test_data.json" # Path for mock, not actually read

        # Create a few dummy documents for testing
        self.sample_documents = [
            Document(text="This is the first test document about apples.", doc_id="doc1", metadata={"category": "fruit"}),
            Document(text="The second document is about bananas and is yellow.", doc_id="doc2", metadata={"category": "fruit"}),
            Document(text="A third document discusses computers and technology.", doc_id="doc3", metadata={"category": "tech"})
        ]
        
        # Reset LlamaIndex Settings before each test to ensure isolation if needed,
        # especially if other tests might modify them globally.
        # For embedding model, initialize_embedding_model handles setting it.
        Settings.llm = None # Reset llm if it was set elsewhere
        Settings.embed_model = None # Reset embed_model

    def tearDown(self):
        """Remove the temporary storage directory after tests."""
        if os.path.exists(self.test_storage_dir):
            shutil.rmtree(self.test_storage_dir)
        # Reset LlamaIndex settings again after tests for cleanliness
        Settings.llm = None
        Settings.embed_model = None

    @patch('data_loader.AnthologyLoader.load_data') # Mock the load_data method
    def test_build_persist_and_load_index(self, mock_load_data: MagicMock):
        """Test the full index building, persisting, and loading process."""
        # Configure the mock to return our sample documents
        mock_load_data.return_value = self.sample_documents

        # 1. Test embedding model initialization (part of build_and_persist_index)
        # initialize_embedding_model() is called inside build_and_persist_index
        # We can check its effect on Settings.embed_model after the call.

        # 2. Build and persist the index using a temporary storage directory
        # Use a specific, known embedding model for testing consistency if desired, or default.
        # The build_and_persist_index function already calls initialize_embedding_model.
        print(f"Test: Building index in {self.test_storage_dir}")
        built_index = build_and_persist_index(
            json_data_path=self.dummy_json_path, # Mocked, so path doesn't matter much here
            storage_dir=self.test_storage_dir,
            embedding_model_name=DEFAULT_EMBED_MODEL_NAME, # Use default for consistency
            force_rebuild=True # Ensure it builds from scratch for the test
        )

        # Assert that load_data was called by the build_index script
        mock_load_data.assert_called_once()

        # Check that the embedding model was set globally by initialize_embedding_model
        self.assertIsNotNone(Settings.embed_model, "Global embed_model should be set.")
        self.assertIsInstance(Settings.embed_model, HuggingFaceEmbedding, "Global embed_model should be a HuggingFaceEmbedding instance.")
        # Check if the model name in the embed_model matches (LlamaIndex might wrap it)
        # This check depends on HuggingFaceEmbedding internal structure, might be fragile
        if hasattr(Settings.embed_model, 'model_name'):
             self.assertEqual(Settings.embed_model.model_name, DEFAULT_EMBED_MODEL_NAME)

        self.assertIsInstance(built_index, VectorStoreIndex, "Should return a VectorStoreIndex instance.")
        self.assertTrue(os.path.exists(os.path.join(self.test_storage_dir, "docstore.json")), "docstore.json should exist after persist.")
        # Check for common default vector store persistence names
        possible_vector_store_files = ["vector_store.json", "default__vector_store.json"]
        vector_store_found = any(
            os.path.exists(os.path.join(self.test_storage_dir, fname))
            for fname in possible_vector_store_files
        )
        if not vector_store_found:
            print(f"Debug: Contents of {self.test_storage_dir}: {os.listdir(self.test_storage_dir)}")
        self.assertTrue(vector_store_found, f"Neither 'vector_store.json' nor 'default__vector_store.json' found in {self.test_storage_dir}. Actual files: {os.listdir(self.test_storage_dir) if os.path.exists(self.test_storage_dir) else 'N/A'}")

        # Add more checks for other persisted files if necessary (e.g., index_store.json)
        self.assertTrue(os.path.exists(os.path.join(self.test_storage_dir, "index_store.json")), "index_store.json should exist.")


        # 3. Test loading the persisted index (implicitly tested if not force_rebuild)
        # To explicitly test loading, call build_and_persist_index again without force_rebuild
        # First, clear the global embed_model to simulate a new run where it needs to be re-initialized by load path
        Settings.embed_model = None 
        print(f"Test: Loading index from {self.test_storage_dir}")
        with patch('builtins.print') as mock_print_load: # To check output messages
            loaded_index = build_and_persist_index(
                json_data_path=self.dummy_json_path, 
                storage_dir=self.test_storage_dir, 
                embedding_model_name=DEFAULT_EMBED_MODEL_NAME,
                force_rebuild=False # This should trigger loading
            )
        
        # Check if the "Found existing index" message was printed
        # This requires checking all calls to print, which can be tricky. A more direct check might be needed.
        # For now, we rely on the fact that load_data would not be called again if loaded from storage.
        # mock_load_data should still have call_count == 1 from the build step.
        self.assertEqual(mock_load_data.call_count, 1, "load_data should not be called again when loading from storage.")

        self.assertIsInstance(loaded_index, VectorStoreIndex, "Loaded index should be a VectorStoreIndex.")
        self.assertIsNotNone(Settings.embed_model, "Global embed_model should be set after loading path too.")
        # Simple check: compare some property or query if possible, but IDs might change if not careful.
        # For now, type checking and existence is primary for this integration test.
        self.assertEqual(len(loaded_index.docstore.docs), len(self.sample_documents), "Loaded index should have the same number of documents.")

if __name__ == "__main__":
    unittest.main() 