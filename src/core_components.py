from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.llms import LLM as LlamaLLM # Generic LLM type for hinting
from llama_index.core.query_engine import BaseQueryEngine, RetrieverQueryEngine # Import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI as LlamaOpenAI # For OpenAI compatible endpoints like OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Optional, TYPE_CHECKING
import os

# Ensure src directory is in Python path for imports if this file is run directly
# (though typically it's imported)
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_loader import AppConfig # Import AppConfig

# if TYPE_CHECKING:
#     from llama_index.core.indices.base import BaseIndex # For type hinting index if needed more generically

# Default local embedding model. This can be made configurable later if needed.
DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_STORAGE_DIR = "./storage"
DEFAULT_SIMILARITY_TOP_K = 3
# Default LLM Temperature if not specified by AppConfig or if needed
DEFAULT_LLM_TEMPERATURE = 0.1 

def initialize_embedding_model(model_name: Optional[str] = None) -> HuggingFaceEmbedding:
    """
    Initializes and sets a HuggingFace sentence-transformer model for embeddings.

    Args:
        model_name (Optional[str]): The name of the HuggingFace model to use.
                                     If None, uses DEFAULT_EMBED_MODEL_NAME.

    Returns:
        HuggingFaceEmbedding: The initialized embedding model.
    """
    if model_name is None:
        model_name = DEFAULT_EMBED_MODEL_NAME
    
    # Check if embedding model is already set and matches to avoid re-initialization
    if isinstance(Settings.embed_model, HuggingFaceEmbedding) and Settings.embed_model.model_name == model_name:
        print(f"Embedding model {model_name} is already initialized and set.")
        return Settings.embed_model
    
    print(f"Initializing embedding model: {model_name}")
    try:
        # Forcing device to CPU for broader compatibility and test stability
        embed_model = HuggingFaceEmbedding(model_name=model_name, device="cpu")
        Settings.embed_model = embed_model
        print(f"Successfully set {model_name} as the global LlamaIndex embedding model (using CPU).")
        return embed_model
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        Settings.embed_model = None # Ensure it's None if initialization fails
        return None

def initialize_llm(config: AppConfig) -> Optional[LlamaLLM]:
    """
    Initializes and sets the LLM based on AppConfig settings.
    Uses OpenAI LLM class configured for OpenRouter.

    Args:
        config (AppConfig): The application configuration object.

    Returns:
        Optional[LlamaLLM]: The initialized LlamaIndex LLM object, or None if API key is missing.
    """
    llm_config = config.get_llm_config()
    # ---- START DEBUG PRINT ----
    print(f"DEBUG initialize_llm: Received llm_config: {llm_config}")
    # ---- END DEBUG PRINT ----

    if not llm_config.get("api_key"):
        print("Warning: OpenRouter API key not found in config. LLM initialization skipped.")
        Settings.llm = None # Explicitly set to None if not configured
        return None

    # Check if LLM is already set and matches config to avoid re-initialization
    # This is a basic check; more complex checks might involve comparing more attributes.
    if isinstance(Settings.llm, LlamaOpenAI):
        current_llm: LlamaOpenAI = Settings.llm
        if (
            current_llm.model == llm_config["model"] and 
            current_llm.api_key == llm_config["api_key"] and 
            str(current_llm.api_base) == llm_config["base_url"]
            # current_llm.temperature == llm_config.get("temperature", DEFAULT_LLM_TEMPERATURE) # If we add temp to config
        ):
            print(f"LLM {llm_config['model']} is already initialized and set with the same configuration.")
            return current_llm

    print(f"Initializing LLM: {llm_config['model']} via OpenRouter at {llm_config['base_url']}")
    try:
        llm = LlamaOpenAI(
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            api_base=llm_config["base_url"],
            temperature=llm_config.get("temperature", DEFAULT_LLM_TEMPERATURE) # Allow temperature override from config if present
        )
        Settings.llm = llm
        print(f"Successfully set {llm.model} as the global LlamaIndex LLM.")
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        Settings.llm = None # Ensure it's None if initialization fails
        return None

def load_anthology_index(
    storage_dir: str = DEFAULT_INDEX_STORAGE_DIR,
    embedding_model_name: Optional[str] = DEFAULT_EMBED_MODEL_NAME
) -> Optional[VectorStoreIndex]:
    """
    Loads the persisted VectorStoreIndex from disk.
    Ensures the embedding model is initialized before loading, as it might be needed by the index.

    Args:
        storage_dir (str): The directory where the index is persisted.
        embedding_model_name (Optional[str]): The name of the embedding model to initialize.
                                              If None, uses the default.

    Returns:
        Optional[VectorStoreIndex]: The loaded index, or None if not found or an error occurs.
    """
    print(f"Attempting to load index from: {storage_dir}")
    if not os.path.exists(os.path.join(storage_dir, "docstore.json")):
        print(f"Index not found in {storage_dir} (docstore.json missing). Please build the index first.")
        return None

    # Ensure embedding model is initialized. This is crucial because the index loading
    # process might require the same embedding model configuration as when it was built.
    initialize_embedding_model(model_name=embedding_model_name)
    
    try:
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        print(f"Successfully loaded index from {storage_dir}.")
        return index
    except Exception as e:
        print(f"Error loading index from {storage_dir}: {e}")
        print("Ensure the index was built correctly and the embedding model matches.")
        return None

def get_anthology_retriever(
    index: VectorStoreIndex, 
    similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K
) -> Optional[BaseRetriever]:
    """
    Creates a retriever from the given VectorStoreIndex.

    Args:
        index (VectorStoreIndex): The loaded LlamaIndex VectorStoreIndex.
        similarity_top_k (int): The number of top similar documents to retrieve.

    Returns:
        Optional[BaseRetriever]: The retriever object, or None if the index is invalid.
    """
    if not isinstance(index, VectorStoreIndex):
        print("Error: Invalid index provided to get_anthology_retriever.")
        return None
    
    print(f"Creating retriever with similarity_top_k={similarity_top_k}")
    return index.as_retriever(similarity_top_k=similarity_top_k)

def get_query_engine(
    index: VectorStoreIndex,
    llm: Optional[LlamaLLM] = None, 
    retriever: Optional[BaseRetriever] = None,
    similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K
) -> Optional[BaseQueryEngine]:
    """
    Creates a query engine from the given VectorStoreIndex.
    Uses specified or global LLM, and specified or new retriever.
    """
    if not isinstance(index, VectorStoreIndex):
        print("Error: Invalid index provided to get_query_engine.")
        return None

    active_llm = llm or Settings.llm
    if active_llm is None:
        print("Warning: LLM not provided/globally set. Query engine may use a mock or fail.")

    active_retriever = retriever
    if active_retriever is None:
        active_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    
    if active_retriever is None:
        print("Error: Failed to obtain a retriever for the query engine.")
        return None

    print(f"Creating query engine...")
    try:
        # Use RetrieverQueryEngine directly for more explicit control
        if active_llm:
            query_engine = RetrieverQueryEngine.from_args(
                retriever=active_retriever,
                llm=active_llm,
                # response_synthesizer can be customized here if needed
            )
        else:
            # This case might not be ideal; a query engine usually needs an LLM for synthesis.
            # However, LlamaIndex might have a default mock or simple synthesizer.
            # For our RAG, an LLM is expected.
            print("Warning: Creating query engine without a specified LLM. May use default/mock behavior.")
            query_engine = RetrieverQueryEngine.from_args(
                retriever=active_retriever
            )
        print("Query engine created successfully.")
        return query_engine
    except Exception as e:
        print(f"Error creating query engine: {e}")
        return None

# Example usage (optional, for direct script testing)
if __name__ == "__main__":
    # Note: For this example to work, you must have already built and persisted an index
    # using `scripts/build_index.py` in the DEFAULT_INDEX_STORAGE_DIR.
    # You also need a valid .env and potentially config.yaml for LLM initialization.

    app_config = AppConfig() # Load configuration

    # 1. Initialize LLM (optional here, as it might be done by a query engine later)
    print("\n--- Testing LLM Initialization ---")
    initialized_llm = initialize_llm(app_config)
    if initialized_llm:
        print(f"LLM initialized: {type(initialized_llm)}, Model: {initialized_llm.model}")
    else:
        print("LLM initialization failed or skipped (e.g., no API key).")

    # 2. Load the index
    print("\n--- Testing Index Loading ---")
    anthology_index = load_anthology_index()

    if anthology_index:
        print("\n--- Testing Retriever Creation ---")
        # 3. Get a retriever
        retriever = get_anthology_retriever(anthology_index, similarity_top_k=2)
        
        if retriever:
            print("Retriever created successfully.")
            # 4. Example: Perform a dummy retrieval (actual queries depend on index content)
            # This requires your dummy index to have some content that can be queried.
            # For a real test, you'd use a query relevant to your actual indexed data.
            sample_query = "language models"
            print(f"\n--- Testing Retrieval for query: '{sample_query}' ---")
            try:
                retrieved_nodes = retriever.retrieve(sample_query)
                if retrieved_nodes:
                    print(f"Retrieved {len(retrieved_nodes)} nodes:")
                    for i, node_with_score in enumerate(retrieved_nodes):
                        print(f"  {i+1}. Node ID: {node_with_score.node.node_id}, Score: {node_with_score.score:.4f}")
                        # print(f"     Text: {node_with_score.node.get_content()[:100]}...") # Deprecated: node.text or node.get_text()
                        print(f"     Text: {node_with_score.node.text[:100]}...")
                else:
                    print("No nodes retrieved for the sample query.")
            except Exception as e:
                print(f"Error during retrieval: {e}")
            
            print("\n--- Testing Query Engine Creation ---")
            if Settings.llm: 
                query_engine = get_query_engine(index=anthology_index, retriever=retriever)
                if query_engine:
                    print("Query engine created successfully.")
                    # Test with a query relevant to potential index content
                    # Modify this query if your actual index content is very different
                    # or if you want to test a specific scenario.
                    # For instance, if your index is small and from test_data_loader, 
                    # a query like "machine learning" or "climate change" might be better.
                    # If it's the full ACL anthology, "transformer models" or "bert" would be good.
                    sample_query_for_engine = "What are some key concepts in transformer models?"
                    print(f"\n--- Testing Query Engine with query: '{sample_query_for_engine}' ---")
                    try:
                        response = query_engine.query(sample_query_for_engine)
                        print(f"\nQuery Engine Response:\n{response}")
                        if hasattr(response, 'source_nodes') and response.source_nodes:
                            print(f"\n--- Source nodes ({len(response.source_nodes)}): ---")
                            for sn_idx, sn in enumerate(response.source_nodes):
                                metadata = sn.node.metadata
                                print(f"  Source {sn_idx+1} (Score: {sn.score:.4f}):")
                                print(f"    Title: {metadata.get('title', 'N/A')}")
                                authors = metadata.get('authors', [])
                                if isinstance(authors, list):
                                    print(f"    Authors: {', '.join(authors) if authors else 'N/A'}")
                                else:
                                    print(f"    Authors: {authors if authors else 'N/A'}")
                                print(f"    Year: {metadata.get('year', 'N/A')}")
                                print(f"    URL: {metadata.get('url', 'N/A')}")
                                # print(f"    Text: {sn.node.text[:100]}...") # First 100 chars of node text
                                print("-" * 20)
                        else:
                            print("No source nodes in response.")
                    except Exception as e:
                        print(f"Error querying with engine: {e}")
                else:
                    print("Failed to create query engine.")
            else:
                print("Skipping query engine test as LLM is not initialized.")
        else:
            print("Failed to create retriever.")
    else:
        print("Failed to load index. Ensure it has been built using scripts/build_index.py.") 