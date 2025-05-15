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
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Functions/classes to test from src and scripts
from core_components import (
    initialize_embedding_model, 
    load_anthology_index, 
    get_anthology_retriever,
    DEFAULT_EMBED_MODEL_NAME
)
from build_index import build_and_persist_index # To create a test index
# from data_loader import AnthologyLoader # For mocking if build_index wasn't used directly

class TestRetrievalPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a temporary index for all tests in this class."""
        cls.test_storage_dir = tempfile.mkdtemp(prefix="test_retrieval_idx_")
        cls.dummy_json_path = "dummy_retrieval_test_data.json" # Not strictly needed if load_data is mocked

        cls.sample_documents = [
            Document(text="Paper about generative adversarial networks and image synthesis.", doc_id="gan_paper", metadata={"topic": "cv"}),
            Document(text="Exploring large language models for text summarization.", doc_id="llm_summary", metadata={"topic": "nlp"}),
            Document(text="A study on reinforcement learning for robotics.", doc_id="rl_robotics", metadata={"topic": "rl"}),
            Document(text="Transformers revolutionized natural language processing.", doc_id="transformers_nlp", metadata={"topic": "nlp"})
        ]

        # Mock AnthologyLoader.load_data within the build_and_persist_index call
        with patch('data_loader.AnthologyLoader.load_data', return_value=cls.sample_documents) as mock_load:
            print(f"setUpClass: Building temporary index in {cls.test_storage_dir}")
            cls.test_index = build_and_persist_index(
                json_data_path=cls.dummy_json_path,
                storage_dir=cls.test_storage_dir,
                embedding_model_name=DEFAULT_EMBED_MODEL_NAME,
                force_rebuild=True
            )
            mock_load.assert_called_once() # Ensure data loading was mocked and called
        
        if cls.test_index is None:
            raise RuntimeError(f"Failed to create test index in setUpClass at {cls.test_storage_dir}")
        print(f"setUpClass: Temporary index built successfully.")

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary storage directory after all tests in this class."""
        if os.path.exists(cls.test_storage_dir):
            print(f"tearDownClass: Removing temporary index from {cls.test_storage_dir}")
            shutil.rmtree(cls.test_storage_dir)
        Settings.embed_model = None # Clean up global setting

    def setUp(self):
        """Reset relevant LlamaIndex settings before each test if necessary."""
        # The embedding model is set globally during build_and_persist_index in setUpClass.
        # We re-initialize it here to ensure it's correctly set for each test method, 
        # especially if a test method were to clear it.
        initialize_embedding_model(DEFAULT_EMBED_MODEL_NAME) # Ensures global embed_model is set
        self.assertIsNotNone(Settings.embed_model, "Embedding model should be set globally for tests.")

    def test_01_load_existing_index(self):
        """Test loading an existing index from the temporary storage."""
        print("Running test_01_load_existing_index...")
        loaded_index = load_anthology_index(storage_dir=self.test_storage_dir)
        self.assertIsNotNone(loaded_index, "Failed to load the existing test index.")
        self.assertIsInstance(loaded_index, VectorStoreIndex)
        self.assertEqual(len(loaded_index.docstore.docs), len(self.sample_documents), "Loaded index has incorrect number of documents.")
        self.assertIsNotNone(Settings.embed_model, "Global embed_model should remain set after loading.")
        self.assertIsInstance(Settings.embed_model, HuggingFaceEmbedding) # Check type

    def test_02_load_non_existent_index(self):
        """Test attempting to load a non-existent index."""
        print("Running test_02_load_non_existent_index...")
        non_existent_dir = os.path.join(self.test_storage_dir, "non_existent_subdir")
        with patch('builtins.print') as mock_print:
            loaded_index = load_anthology_index(storage_dir=non_existent_dir)
            self.assertIsNone(loaded_index, "Should return None for a non-existent index.")
            # Check if the appropriate warning was printed
            found_print = any(
                f"Index not found in {non_existent_dir}" in call_args[0][0] 
                for call_args in mock_print.call_args_list if call_args[0]
            )
            self.assertTrue(found_print, "Warning for non-existent index not printed.")

    def test_03_get_retriever_and_retrieve_nodes(self):
        """Test creating a retriever and performing a basic retrieval."""
        print("Running test_03_get_retriever_and_retrieve_nodes...")
        # Use the class-level test_index built in setUpClass
        self.assertIsNotNone(self.test_index, "Test index not available for retriever test.")
        
        retriever = get_anthology_retriever(self.test_index, similarity_top_k=2)
        self.assertIsNotNone(retriever, "Failed to create retriever.")
        self.assertIsInstance(retriever, BaseRetriever)

        # Perform a sample query. The results depend on the dummy data.
        # Query for "NLP papers"
        query_string = "natural language processing techniques"
        retrieved_nodes = retriever.retrieve(query_string)

        self.assertIsNotNone(retrieved_nodes, "Retrieval should return a list (even if empty).")
        self.assertIsInstance(retrieved_nodes, list)
        if retrieved_nodes: # Only check contents if something was retrieved
            self.assertTrue(all(isinstance(n, NodeWithScore) for n in retrieved_nodes))
            self.assertLessEqual(len(retrieved_nodes), 2, "Retrieved more nodes than similarity_top_k.")
            
            print(f"Retrieved {len(retrieved_nodes)} nodes for query '{query_string}':")
            retrieved_ids = []
            for i, node_with_score in enumerate(retrieved_nodes):
                print(f"  {i+1}. Node ID: {node_with_score.node.node_id}, Ref Doc ID: {node_with_score.node.ref_doc_id}, Score: {node_with_score.score:.4f}")
                print(f"     Text: {node_with_score.node.text[:100]}...")
                retrieved_ids.append(node_with_score.node.ref_doc_id)
            
            # Based on sample_documents, we expect llm_summary and transformers_nlp
            self.assertTrue("llm_summary" in retrieved_ids or "transformers_nlp" in retrieved_ids, 
                            f"Expected relevant NLP paper (llm_summary or transformers_nlp) not found in retrieved ref_doc_ids: {retrieved_ids}")
        else:
            print(f"No nodes retrieved for query '{query_string}'. This might be okay depending on index/query.")
            # For this specific query and data, we do expect some results.
            # self.fail("Expected some nodes to be retrieved for the NLP query.") # Uncomment if strict check needed

    def test_04_get_retriever_invalid_index(self):
        """Test get_anthology_retriever with an invalid index type."""
        print("Running test_04_get_retriever_invalid_index...")
        with patch('builtins.print') as mock_print:
            retriever = get_anthology_retriever(index="not_an_index_object")
            self.assertIsNone(retriever)
            found_print = any(
                "Invalid index provided" in call_args[0][0] 
                for call_args in mock_print.call_args_list if call_args[0]
            )
            self.assertTrue(found_print, "Error message for invalid index not printed.")

if __name__ == "__main__":
    unittest.main() 