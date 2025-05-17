import pytest
import tempfile
import shutil
import os
import yaml

from src.config_loader import AppConfig, IndexBuilderConfig, QueryEngineBuilderConfig
from src.index_builder import IndexBuilder
from src.query_engine_builder import QueryEngineBuilder
from llama_index.core.base.response.schema import Response
from llama_index.core import Settings

# Fixture for a temporary directory for index storage
@pytest.fixture
def temp_index_storage_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_integration_idx_")
    yield temp_dir
    shutil.rmtree(temp_dir)

# Fixture to reset LlamaIndex global settings before each test
@pytest.fixture(autouse=True)
def reset_llama_index_settings_integration():
    # Before test
    Settings.embed_model = None
    Settings.node_parser = None
    yield
    # After test
    Settings.embed_model = None
    Settings.node_parser = None

def test_build_and_query_end_to_end(temp_index_storage_dir):
    """Test building an index and querying it end-to-end."""
    dummy_config_path = "tests/dummy_config.yaml"

    # Load and update the dummy config with the temporary storage directory
    with open(dummy_config_path, 'r') as f:
        raw_config_content = yaml.safe_load(f)
    
    raw_config_content['index_builder']['storage_dir'] = temp_index_storage_dir

    # Create a temporary modified config file
    temp_config_file_path = os.path.join(temp_index_storage_dir, "temp_integration_config.yaml")
    with open(temp_config_file_path, 'w') as f:
        yaml.dump(raw_config_content, f)

    # Initialize AppConfig with the path to the temporary config file
    app_config = AppConfig(config_path=temp_config_file_path)
    
    index_builder_cfg = app_config.get_index_builder_config()
    query_engine_builder_cfg = app_config.get_query_engine_builder_config()

    assert index_builder_cfg.storage_dir == temp_index_storage_dir
    assert index_builder_cfg.corpus_path == "tests/dummy_corpus.json"

    # 1. Build the Index
    print(f"Building index in: {index_builder_cfg.storage_dir}")
    idx_builder = IndexBuilder(config=index_builder_cfg)
    # Force rebuild to ensure it uses the dummy corpus and fresh settings
    index = idx_builder.build(force_rebuild=True) 
    assert index is not None
    print("Index built successfully.")

    # Check if index files were created
    assert os.path.exists(os.path.join(index_builder_cfg.storage_dir, index_builder_cfg.docstore_filename))
    # For the default SimpleVectorStore, LlamaIndex persists it as default__vector_store.json
    expected_vector_store_filename = "default__vector_store.json" 
    assert os.path.exists(os.path.join(index_builder_cfg.storage_dir, expected_vector_store_filename))
    assert os.path.exists(os.path.join(index_builder_cfg.storage_dir, index_builder_cfg.index_store_filename))
    print("Index files found on disk.")

    # 2. Build the Query Engine
    engine_builder = QueryEngineBuilder(index=index, config=query_engine_builder_cfg)
    query_engine = engine_builder.build()
    assert query_engine is not None
    print("Query engine built successfully.")

    # 3. Query the Index
    query_text = "fox"
    response = query_engine.query(query_text)

    assert response is not None
    assert isinstance(response, Response)
    assert response.response # Check that the response text is not empty
    print(f"Query: '{query_text}'")
    print(f"Response: '{response.response[:100]}...'") # Print truncated response
    assert len(response.source_nodes) > 0
    print(f"Retrieved {len(response.source_nodes)} source nodes.")

    # A simple check on source node content
    found_fox_in_source = False
    for node in response.source_nodes:
        if "fox" in node.get_content().lower():
            found_fox_in_source = True
            print(f"Found 'fox' in source node: {node.node_id}")
            break
    assert found_fox_in_source 