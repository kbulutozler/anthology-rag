#!/usr/bin/env python3
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Ensure src directory is in Python path - This line is now redundant after adding project root
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_loader import AppConfig
from src.core_components import (
    initialize_embedding_model,
    initialize_llm,
    load_anthology_index,
    get_anthology_retriever,
    get_query_engine,
    DEFAULT_EMBED_MODEL_NAME,
    DEFAULT_INDEX_STORAGE_DIR,
    DEFAULT_SIMILARITY_TOP_K
)
from llama_index.core.settings import Settings

def main():
    """
    Main function to run the ACL Anthology RAG chat CLI.
    """
    print("--- ACL Anthology RAG Chat Assistant ---")

    # 1. Load Configuration
    print("\nLoading configurations...")
    app_config = AppConfig()
    if not app_config.openrouter_api_key:
        print("ERROR: OpenRouter API key is not configured. Please set it in your .env file.")
        print("Chat CLI cannot start without an API key.")
        return
    print("Configurations loaded.")

    # 2. Initialize Embedding Model
    print("\nInitializing embedding model...")
    embed_model = initialize_embedding_model(model_name=DEFAULT_EMBED_MODEL_NAME)
    if not embed_model:
        print("ERROR: Failed to initialize the embedding model.")
        return
    print(f"Embedding model '{embed_model.model_name}' initialized.")

    # 3. Initialize LLM
    print("\nInitializing Large Language Model...")
    llm = initialize_llm(app_config)
    if not llm:
        print("ERROR: Failed to initialize the LLM. Check API key and OpenRouter settings in config.yaml/.env.")
        return
    print(f"LLM '{llm.model}' initialized via {app_config.openrouter_base_url}.")

    # 4. Load Anthology Index
    print("\nLoading ACL Anthology index...")
    # Ensure embedding model is set globally before loading the index
    if Settings.embed_model is None:
         print("ERROR: Embedding model is not initialized. Cannot load index.")
         # This case should ideally not happen if initialize_embedding_model succeeds, but as a safeguard:
         return 

    index = load_anthology_index(storage_dir=DEFAULT_INDEX_STORAGE_DIR, embedding_model_name=embed_model.model_name)
    
    if index is None:
        print(f"ERROR: Failed to load the index from '{DEFAULT_INDEX_STORAGE_DIR}'.")
        print("Please ensure the index has been built using 'python scripts/build_index.py'.")
        return
    print("Index loaded successfully.")

    # 5. Create Retriever
    print("\nCreating retriever...")
    retriever = get_anthology_retriever(index, similarity_top_k=DEFAULT_SIMILARITY_TOP_K)
    if retriever is None:
        print("ERROR: Failed to create retriever.")
        return
    print("Retriever created successfully.")

    # 6. Create Query Engine
    print("\nCreating query engine...")
    query_engine = get_query_engine(index=index, llm=llm, retriever=retriever)
    if query_engine is None:
        print("ERROR: Failed to create query engine.")
        return
    print("Query engine created successfully.")

    # 7. Start Chat Loop
    print("\nSystem ready. Type your query or 'quit' to exit.")
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ['quit', 'exit']:
                break
            if not query.strip():
                continue # Skip empty queries

            print("\nThinking...")
            response = query_engine.query(query)
            
            print("\nAssistant:")
            print(response)

            if hasattr(response, 'source_nodes') and response.source_nodes:
                print("\n--- Sources ---")
                for i, sn in enumerate(response.source_nodes):
                    metadata = sn.node.metadata
                    print(f"Source {i+1} (Score: {sn.score:.4f}):")
                    print(f"  Title: {metadata.get('title', 'N/A')}")
                    authors = metadata.get('authors', [])
                    if isinstance(authors, list):
                         print(f"  Authors: {", ".join(authors) if authors else "N/A"}")
                    else:
                         print(f"  Authors: {authors if authors else 'N/A'}")
                    print(f"  Year: {metadata.get('year', 'N/A')}")
                    print(f"  URL: {metadata.get('url', 'N/A')}")
                    # print(f"  Text: {sn.node.text[:100]}...") # Optional: show snippet
                    print("-------------------------------")
                print("-" * 80)

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break
        except Exception as e:
            print(f"An error occurred during the chat session: {e}")
            # Continue the loop or break based on desired error handling

    print("--- Chat session ended. ---")

if __name__ == "__main__":
    main() 