from typing import Optional

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine

from src.config_loader import QueryEngineBuilderConfig
from src.core_components import initialize_hf_embedding_model

class QueryEngineBuilder:
    """
    Builds a query engine from a VectorStoreIndex and configuration.
    """
    def __init__(self, index: VectorStoreIndex, config: QueryEngineBuilderConfig):
        """
        Args:
            index (VectorStoreIndex): The VectorStoreIndex to query.
            config (QueryEngineBuilderConfig): Configuration for the query engine.
        """
        if not isinstance(index, VectorStoreIndex):
            raise TypeError("index must be an instance of VectorStoreIndex")
        if not isinstance(config, QueryEngineBuilderConfig):
            raise TypeError("config must be an instance of QueryEngineBuilderConfig")

        self.index = index
        self.config = config

    def build(self) -> BaseQueryEngine:
        """
        Configures LlamaIndex global settings and builds the query engine.

        Returns:
            BaseQueryEngine: The configured query engine.
        """
        # Ensure the correct embedding model is set globally for LlamaIndex.
        # This is crucial if the index was loaded from storage without an active embed model in Settings,
        # or if a different model is desired for querying (though typically they should match).
        print(f"QueryEngineBuilder: Setting embed model in LlamaIndex.Settings: {self.config.embedding_model_name}")
        initialize_hf_embedding_model(model_name=self.config.embedding_model_name)

        # Set chunk size and overlap from config for consistency, though these primarily affect indexing.
        # If the index is already built, these might not have a direct effect on a standard query engine
        # unless a retriever using these specific settings is being configured here.
        # For a basic VectorIndexQueryEngine, these are not directly used at query time from Settings,
        # but it's good practice to have them consistent with the index build process.
        if self.config.chunk_size is not None:
            # Settings.chunk_size = self.config.chunk_size # Deprecated
            # Settings.node_parser has chunk_size, but node_parser is usually set during indexing.
            # For querying, these are less critical unless re-parsing or specific retriever.
            pass 
        if self.config.chunk_overlap is not None:
            # Settings.chunk_overlap = self.config.chunk_overlap # Deprecated
            pass

        print(f"QueryEngineBuilder: Building query engine with similarity_top_k={self.config.similarity_top_k}")
        query_engine = self.index.as_query_engine(
            similarity_top_k=self.config.similarity_top_k
        )
        print("QueryEngineBuilder: Query engine built successfully.")
        return query_engine

if __name__ == "__main__":
    # This is a placeholder for potential direct testing of QueryEngineBuilder.
    # To run this, you would need:
    # 1. A built and persisted VectorStoreIndex.
    # 2. A QueryEngineBuilderConfig instance.
    # Example (conceptual):
    # from src.config_loader import AppConfig
    # from src.index_builder import IndexBuilder
    
    # print("Conceptual test for QueryEngineBuilder...")
    # # Load app configuration
    # app_config = AppConfig(config_path='config.yaml') # Assumes a valid config.yaml exists
    # index_builder_config = app_config.get_index_builder_config()
    # query_engine_config = app_config.get_query_engine_builder_config()
    
    # # Ensure storage directory exists or create a dummy one for a test
    # if not os.path.exists(index_builder_config.storage_dir):
    #     print(f"Storage directory {index_builder_config.storage_dir} not found. Cannot load index.")
    #     print("Please build an index first using index_builder.py or ensure config.yaml points to a valid index.")
    # else:
    #     # Build/Load the index
    #     print("Loading index...")
    #     idx_builder = IndexBuilder(config=index_builder_config)
    #     try:
    #         index = idx_builder.load() # Or build if not present: idx_builder.build(force_rebuild=False)
    #         print("Index loaded.")

    #         # Build the query engine
    #         print("Building query engine...")
    #         engine_builder = QueryEngineBuilder(index=index, config=query_engine_config)
    #         query_engine = engine_builder.build()
    #         print("Query engine built.")

    #         # Perform a query
    #         test_query = "What is sustainable software engineering?"
    #         print(f"Performing test query: '{test_query}'")
    #         response = query_engine.query(test_query)
    #         print(f"Response: {response}")
    #         if response.source_nodes:
    #             print("Source nodes:")
    #             for i, node in enumerate(response.source_nodes):
    #                 print(f"  Node {i+1}: Score = {node.score:.4f}, Text = \"{node.text[:100]}...\"")
    #     except Exception as e:
    #         print(f"An error occurred during the test: {e}")
    pass 