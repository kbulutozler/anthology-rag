import os
import unittest
import shutil
import tempfile
from unittest.mock import patch, MagicMock, call

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter # Used for isinstance check
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.index_builder import IndexBuilder
from src.config_loader import IndexBuilderConfig
# DocumentLoader and initialize_hf_embedding_model are dependencies of IndexBuilder,
# so they will be mocked where necessary.

import pytest

# Fixture for temporary directory, replacing setUp/tearDown for directory creation/cleanup
@pytest.fixture
def test_storage_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_idx_builder_")
    yield temp_dir
    shutil.rmtree(temp_dir)

# Fixture for mock documents
@pytest.fixture
def mock_documents():
    return [Document(text="Test doc 1"), Document(text="Test doc 2")]

# Fixture for IndexBuilderConfig
@pytest.fixture
def index_builder_config(test_storage_dir):
    return IndexBuilderConfig(
        storage_dir=test_storage_dir,
        embedding_model_name="sentence-transformers/test-model",
        corpus_path="dummy/corpus.json",
        corpus_id_field="id",
        corpus_text_fields=["text"],
        corpus_metadata_fields=["meta"],
        chunk_size=100,
        chunk_overlap=10,
        docstore_filename="test_docstore.json",
        vector_store_filename="test_vector_store.json",
        index_store_filename="test_index_store.json"
    )

# Fixture to reset LlamaIndex global settings before each test
@pytest.fixture(autouse=True)
def reset_llama_index_settings():
    # Before test
    Settings.embed_model = None
    Settings.node_parser = None
    yield
    # After test
    Settings.embed_model = None
    Settings.node_parser = None

# No need for a class inheriting from unittest.TestCase in pytest

@patch('src.index_builder.initialize_hf_embedding_model')
@patch('src.index_builder.DocumentLoader')
@patch('llama_index.core.VectorStoreIndex.from_documents')
@patch('src.index_builder.IndexBuilder.persist') # Mock instance method
@patch('os.path.exists') # Keep the mock
def test_build_new_index_no_documents_provided(mock_os_path_exists, mock_persist, mock_from_documents, MockDocumentLoader, mock_init_embed, index_builder_config, mock_documents):
    """Test building a new index when no documents are passed and index doesn't exist on disk."""
    # Only control the *first* call to os.path.exists which checks for the docstore
    mock_os_path_exists.return_value = False # Simulate index not existing initially

    mock_init_embed.return_value = MagicMock(spec=HuggingFaceEmbedding)

    mock_doc_loader_instance = MockDocumentLoader.return_value
    mock_doc_loader_instance.load_data.return_value = mock_documents

    mock_index_instance = MagicMock(spec=VectorStoreIndex)
    mock_from_documents.return_value = mock_index_instance

    builder = IndexBuilder(config=index_builder_config)
    index = builder.build()

    # Assert that os.path.exists was called at least once with the target docstore path
    target_docstore_path = os.path.join(index_builder_config.storage_dir, index_builder_config.docstore_filename)
    assert call(target_docstore_path) in mock_os_path_exists.call_args_list
    assert len(mock_os_path_exists.call_args_list) >= 1 # Ensure it was called at all

    mock_init_embed.assert_called_once_with(model_name=index_builder_config.embedding_model_name)
    assert Settings.node_parser is not None
    assert isinstance(Settings.node_parser, SentenceSplitter)
    assert Settings.node_parser.chunk_size == index_builder_config.chunk_size
    MockDocumentLoader.assert_called_once_with(
        corpus_path=index_builder_config.corpus_path,
        text_fields=index_builder_config.corpus_text_fields,
        metadata_fields=index_builder_config.corpus_metadata_fields,
        id_field=index_builder_config.corpus_id_field
    )
    mock_doc_loader_instance.load_data.assert_called_once()
    mock_from_documents.assert_called_once_with(mock_documents, show_progress=True)
    assert index is mock_index_instance
    assert builder.index is mock_index_instance
    mock_persist.assert_called_once() # Check that persist was called on the builder instance

@patch('src.index_builder.initialize_hf_embedding_model')
@patch('llama_index.core.VectorStoreIndex.from_documents')
@patch('src.index_builder.IndexBuilder.persist')
@patch('os.path.exists')
def test_build_new_index_with_documents_provided(mock_os_path_exists, mock_persist, mock_from_documents, mock_init_embed, index_builder_config, mock_documents):
    """Test building a new index when documents are directly provided."""
    # Only control the *first* call to os.path.exists
    mock_os_path_exists.return_value = False # Simulate index not existing initially

    mock_init_embed.return_value = MagicMock(spec=HuggingFaceEmbedding)
    mock_index_instance = MagicMock(spec=VectorStoreIndex)
    mock_from_documents.return_value = mock_index_instance

    builder = IndexBuilder(config=index_builder_config)
    index = builder.build(documents=mock_documents)

    # Assert that os.path.exists was called at least once with the target docstore path
    target_docstore_path = os.path.join(index_builder_config.storage_dir, index_builder_config.docstore_filename)
    assert call(target_docstore_path) in mock_os_path_exists.call_args_list

    mock_init_embed.assert_called_once_with(model_name=index_builder_config.embedding_model_name)
    mock_from_documents.assert_called_once_with(mock_documents, show_progress=True)
    assert index is mock_index_instance
    mock_persist.assert_called_once()

@patch('src.index_builder.initialize_hf_embedding_model')
@patch('src.index_builder.load_index_from_storage')
@patch('llama_index.core.storage.storage_context.StorageContext.from_defaults') # Corrected mock target
@patch('os.path.exists') # Keep the mock
def test_load_existing_index(mock_os_path_exists, mock_storage_context_from_defaults, mock_load_idx_from_storage, mock_init_embed, index_builder_config):
    """Test loading an index when it exists on disk and force_rebuild is False."""
    # Only control the *first* call to os.path.exists
    mock_os_path_exists.return_value = True # Simulate index existing initially

    mock_init_embed.return_value = MagicMock(spec=HuggingFaceEmbedding)
    mock_storage_context_instance = MagicMock()
    # Corrected call with only persist_dir
    mock_storage_context_from_defaults.return_value = mock_storage_context_instance
    mock_loaded_index_instance = MagicMock(spec=VectorStoreIndex)
    mock_load_idx_from_storage.return_value = mock_loaded_index_instance

    builder = IndexBuilder(config=index_builder_config)
    index = builder.build(force_rebuild=False) # Default is False

    # Assert that os.path.exists was called at least once with the target docstore path
    target_docstore_path = os.path.join(index_builder_config.storage_dir, index_builder_config.docstore_filename)
    assert call(target_docstore_path) in mock_os_path_exists.call_args_list

    # build() calls self.load() in this case
    mock_init_embed.assert_called_once_with(model_name=index_builder_config.embedding_model_name)
    assert Settings.node_parser is not None
    assert isinstance(Settings.node_parser, SentenceSplitter)
    # Corrected assertion for from_defaults call
    mock_storage_context_from_defaults.assert_called_once_with(
        persist_dir=index_builder_config.storage_dir
    )
    mock_load_idx_from_storage.assert_called_once_with(mock_storage_context_instance)
    assert index is mock_loaded_index_instance
    assert builder.index is mock_loaded_index_instance

@patch('src.index_builder.initialize_hf_embedding_model')
@patch('src.index_builder.DocumentLoader')
@patch('llama_index.core.VectorStoreIndex.from_documents')
@patch('src.index_builder.IndexBuilder.persist')
@patch('os.path.exists') # Keep the mock
def test_build_force_rebuild(mock_os_path_exists, mock_persist, mock_from_documents, MockDocumentLoader, mock_init_embed, index_builder_config, mock_documents):
    """Test building an index with force_rebuild=True even if it exists."""
    # Only control the *first* call to os.path.exists
    mock_os_path_exists.return_value = True # Simulate index existing initially

    mock_init_embed.return_value = MagicMock(spec=HuggingFaceEmbedding)
    mock_doc_loader_instance = MockDocumentLoader.return_value
    mock_doc_loader_instance.load_data.return_value = mock_documents
    mock_index_instance = MagicMock(spec=VectorStoreIndex)
    mock_from_documents.return_value = mock_index_instance

    builder = IndexBuilder(config=index_builder_config)
    index = builder.build(force_rebuild=True)

    # Assert that os.path.exists was called at least once with the target docstore path
    target_docstore_path = os.path.join(index_builder_config.storage_dir, index_builder_config.docstore_filename)
    assert call(target_docstore_path) in mock_os_path_exists.call_args_list

    # Even though exists, should proceed with build steps
    mock_init_embed.assert_called_once_with(model_name=index_builder_config.embedding_model_name)
    MockDocumentLoader.assert_called_once()
    mock_doc_loader_instance.load_data.assert_called_once()
    mock_from_documents.assert_called_once_with(mock_documents, show_progress=True)
    assert index is mock_index_instance
    mock_persist.assert_called_once()

def test_persist_no_index(index_builder_config):
    """Test persist raises RuntimeError if index is not built."""
    builder = IndexBuilder(config=index_builder_config)
    with pytest.raises(RuntimeError, match="No index to persist. Build or load the index first."):
        builder.persist()

@patch('llama_index.core.storage.storage_context.StorageContext.persist')
def test_persist_success(mock_storage_persist, index_builder_config):
    """Test successful persistence call."""
    builder = IndexBuilder(config=index_builder_config)
    builder.index = MagicMock(spec=VectorStoreIndex)
    builder.index.storage_context = MagicMock() # Mock the storage_context attribute of the mock index
    builder.index.storage_context.persist = mock_storage_persist # Assign the mock to the method we want to check

    builder.persist()

    # Corrected assertion for persist call
    mock_storage_persist.assert_called_once_with(
        persist_dir=index_builder_config.storage_dir
    )

def test_get_index_not_built(index_builder_config):
    """Test get_index raises RuntimeError if index is not available."""
    builder = IndexBuilder(config=index_builder_config)
    with pytest.raises(RuntimeError, match="Index not built or loaded yet."):
        builder.get_index()

def test_get_index_success(index_builder_config):
    """Test get_index returns the index."""
    builder = IndexBuilder(config=index_builder_config)
    mock_index = MagicMock(spec=VectorStoreIndex)
    builder.index = mock_index
    assert builder.get_index() is mock_index

@patch('src.index_builder.os.makedirs') # Mock os.makedirs
@patch('src.index_builder.initialize_hf_embedding_model')
@patch('src.index_builder.DocumentLoader')
@patch('llama_index.core.VectorStoreIndex.from_documents')
@patch('src.index_builder.IndexBuilder.persist') # Mock instance method
@patch('os.path.exists') # Keep the mock
@patch('llama_index.core.storage.storage_context.StorageContext.from_defaults') # Mock StorageContext.from_defaults
@patch('src.index_builder.load_index_from_storage') # Mock load_index_from_storage as it's called by load()
def test_build_creates_storage_dir(
    mock_load_idx_from_storage,             # Corresponds to @patch('src.index_builder.load_index_from_storage')
    mock_storage_context_from_defaults,     # Corresponds to @patch('llama_index.core.storage.storage_context.StorageContext.from_defaults')
    mock_os_path_exists,                    # Corresponds to @patch('os.path.exists')
    mock_persist,                           # Corresponds to @patch('src.index_builder.IndexBuilder.persist')
    mock_from_documents,                    # Corresponds to @patch('llama_index.core.VectorStoreIndex.from_documents')
    MockDocumentLoader,                     # Corresponds to @patch('src.index_builder.DocumentLoader')
    mock_init_embed,                        # Corresponds to @patch('src.index_builder.initialize_hf_embedding_model')
    mock_os_makedirs,                       # Corresponds to @patch('src.index_builder.os.makedirs')
    index_builder_config,
    mock_documents
):
    """Test that build() calls os.makedirs for the storage directory."""
    # We don't need to control os.path.exists here, but we keep the mock
    # because it's in the decorator list. We just won't configure its side_effect.

    # Configure os.path.exists to return False for the docstore check
    target_docstore_path = os.path.join(index_builder_config.storage_dir, index_builder_config.docstore_filename)
    mock_os_path_exists.return_value = False
    
    # Mock the StorageContext.from_defaults and load_index_from_storage calls that happen if load() is triggered
    mock_storage_context_from_defaults.return_value = MagicMock()
    mock_load_idx_from_storage.return_value = MagicMock(spec=VectorStoreIndex)

    mock_init_embed.return_value = MagicMock(spec=HuggingFaceEmbedding)
    MockDocumentLoader.return_value.load_data.return_value = mock_documents
    mock_from_documents.return_value = MagicMock(spec=VectorStoreIndex)

    builder = IndexBuilder(config=index_builder_config)
    # Call build with force_rebuild=True to ensure it takes the build path explicitly
    # This makes the test clearer about its intent and avoids reliance on the initial os.path.exists mock.
    builder.build(force_rebuild=True)

    mock_os_makedirs.assert_called_once_with(index_builder_config.storage_dir, exist_ok=True)
    
    # Also assert that the load path (StorageContext.from_defaults and load_index_from_storage) was NOT called
    mock_storage_context_from_defaults.assert_not_called()
    mock_load_idx_from_storage.assert_not_called()

if __name__ == "__main__":
    unittest.main() 