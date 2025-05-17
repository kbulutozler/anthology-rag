#!/usr/bin/env python3
import os
import sys
import time

# Add project root to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.index_builder import build_and_persist_index, DEFAULT_JSON_DATA_PATH, DEFAULT_STORAGE_DIR, EMBEDDING_MODEL_NAME

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and persist the ACL Anthology VectorStoreIndex.")
    parser.add_argument("--rebuild", action="store_true", help="Force a rebuild of the index, even if storage exists.")

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