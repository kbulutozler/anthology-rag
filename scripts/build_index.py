#!/usr/bin/env python3
import os
import sys
import time

# Add project root to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# The previous line adding ../src might still be useful depending on exact execution context
# but adding the project root covers imports like from src.config_loader
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(current_dir, '../src')) # This line can likely be removed or commented out

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.settings import Settings # For type hinting or direct access if needed
from llama_index.core.node_parser import SentenceSplitter # Import SentenceSplitter

from src.data_loader import AnthologyLoader # Now imports correctly from src
from src.core_components import initialize_embedding_model # Now imports correctly from src

# --- Configuration ---
# Directory to store the persisted index
DEFAULT_STORAGE_DIR = "./storage"
# Path to the JSON data file
DEFAULT_JSON_DATA_PATH = "data/anthology+abstracts.json"
# Embedding model (can be overridden if needed, but core_components has a default)
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

def build_and_persist_index(
    json_data_path: str = DEFAULT_JSON_DATA_PATH,
    storage_dir: str = DEFAULT_STORAGE_DIR,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    force_rebuild: bool = False
):
    """
    Builds a VectorStoreIndex from the ACL Anthology data and persists it to disk.

    Args:
        json_data_path (str): Path to the input JSON data file.
        storage_dir (str): Directory to save the index to.
        embedding_model_name (str): Name of the embedding model to use.
        force_rebuild (bool): If True, rebuild the index even if storage exists.
    """
    # Ensure the storage directory exists
    os.makedirs(storage_dir, exist_ok=True)

    # Check if index already exists unless force_rebuild is True
    index_exists = os.path.exists(os.path.join(storage_dir, "docstore.json"))
    if index_exists and not force_rebuild:
        print(f"Index already exists in {storage_dir}. Loading existing index.")
        try:
            # Ensure embedding model is initialized before loading
            initialize_embedding_model(model_name=embedding_model_name)
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(storage_context)
            print("Successfully loaded existing index.")
            return index
        except Exception as e:
            print(f"Error loading existing index: {e}. Attempting to rebuild.")
            # If loading fails, fall through to rebuild
    
    print("--- Starting Index Building Process ---")
    print(f"Using JSON data from: {json_data_path}")
    print(f"Index storage directory: {storage_dir}")
    print(f"Embedding model specified: {embedding_model_name}")

    # 1. Initialize embedding model
    initialize_embedding_model(model_name=embedding_model_name)

    # 2. Configure Node Parser with increased chunk size
    print("Step 2: Configuring Node Parser...")
    Settings.node_parser = SentenceSplitter(
        chunk_size=2048, # Increased chunk size
        chunk_overlap=200 # Keep a reasonable overlap
    )
    print(f"Node parser configured with chunk size: {Settings.node_parser.chunk_size}, overlap: {Settings.node_parser.chunk_overlap}\n")

    # 3. Load data
    print("Step 3: Loading documents...") # Adjusted step number
    loader = AnthologyLoader(json_data_path)
    documents = loader.load_data()
    print(f"Loaded {len(documents)} documents.\n")

    if not documents:
        print("No documents loaded. Cannot build index.")
        return None

    # 4. Create VectorStoreIndex
    print("Step 4: Creating VectorStoreIndex...") # Adjusted step number
    print("This might take a while depending on the number of documents and the embedding model...")
    # The node parser and embedding model are automatically used because they are set globally in Settings
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    print("VectorStoreIndex created successfully.\n")

    # 5. Persist index
    print("Step 5: Persisting index to disk...")
    index.storage_context.persist(persist_dir=storage_dir)
    print(f"Index persisted to {storage_dir}.\n")

    print("--- Index Building and Persistence Complete ---")
    return index

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and persist the ACL Anthology VectorStoreIndex.")
    parser.add_argument("--rebuild", action="store_true", help="Force a rebuild of the index, even if storage exists.")
    # Add arguments for custom paths or model names if needed later
    # parser.add_argument("--data", type=str, default=DEFAULT_JSON_DATA_PATH, help="Path to the input JSON data file.")
    # parser.add_argument("--storage", type=str, default=DEFAULT_STORAGE_DIR, help="Directory to save/load the index from.")
    # parser.add_argument("--model", type=str, default=EMBEDDING_MODEL_NAME, help="Embedding model name.")

    args = parser.parse_args()

    start_time = time.time()
    print(f"Current working directory: {os.getcwd()}") # Debug print

    build_and_persist_index(
        json_data_path=DEFAULT_JSON_DATA_PATH,
        storage_dir=DEFAULT_STORAGE_DIR,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        force_rebuild=args.rebuild
    )
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds") 