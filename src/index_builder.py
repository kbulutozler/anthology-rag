import os
import time
from typing import Optional, List
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Document
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SentenceSplitter
from src.document_loader import DocumentLoader
from src.core_components import initialize_hf_embedding_model
from src.config_loader import IndexBuilderConfig

class IndexBuilder:
    """
    Encapsulates the lifecycle and operations of a VectorStoreIndex for a generic document collection.
    Provides methods to build, load, persist, and access the index.

    Args:
        config (IndexBuilderConfig): Configuration object containing all settings for the index builder.
        node_parser (Optional[SentenceSplitter]): Optional pre-configured node parser. If None,
            one will be created using chunk_size and chunk_overlap from the config.
    """
    def __init__(
        self,
        config: IndexBuilderConfig,
        node_parser: Optional[SentenceSplitter] = None
    ):
        self.config = config
        self.storage_dir = config.storage_dir
        self.embedding_model_name = config.embedding_model_name
        self.corpus_path = config.corpus_path
        self.corpus_id_field = config.corpus_id_field
        self.corpus_text_fields = config.corpus_text_fields
        self.corpus_metadata_fields = config.corpus_metadata_fields
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.docstore_filename = config.docstore_filename
        self.vector_store_filename = config.vector_store_filename
        self.index_store_filename = config.index_store_filename
        
        self.node_parser = node_parser or SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.index: Optional[VectorStoreIndex] = None

    def build(self, documents: Optional[List[Document]] = None, force_rebuild: bool = False) -> VectorStoreIndex:
        """
        Build the index from a list of documents, or load them from corpus_path if not provided.
        Persists the index to disk.
        Args:
            documents (Optional[List[Document]]): Documents to index. If None, loads from corpus_path.
            force_rebuild (bool): If True, rebuild even if index exists on disk.
        Returns:
            VectorStoreIndex: The built index.
        """
        os.makedirs(self.storage_dir, exist_ok=True)
        index_exists = os.path.exists(os.path.join(self.storage_dir, self.docstore_filename))
        if index_exists and not force_rebuild:
            return self.load()
        
        print("--- Starting Index Building Process ---")
        print(f"Index storage directory: {self.storage_dir}")
        print(f"Embedding model specified: {self.embedding_model_name}")
        initialize_hf_embedding_model(model_name=self.embedding_model_name)
        Settings.node_parser = self.node_parser
        print(f"Node parser configured with chunk size: {self.node_parser.chunk_size}, overlap: {self.node_parser.chunk_overlap}")
        if documents is None:
            if not self.corpus_path:
                raise ValueError("No documents provided and no corpus_path set.")
            print(f"Loading documents from: {self.corpus_path}")
            loader = DocumentLoader(
                corpus_path=self.corpus_path,
                text_fields=self.corpus_text_fields,
                metadata_fields=self.corpus_metadata_fields,
                id_field=self.corpus_id_field
            )
            documents = loader.load_data()
        print(f"Loaded {len(documents)} documents.")
        if not documents:
            raise ValueError("No documents loaded. Cannot build index.")
        print("Creating VectorStoreIndex (this may take a while)...")
        self.index = VectorStoreIndex.from_documents(documents, show_progress=True)
        print("VectorStoreIndex created successfully.")
        self.persist()
        return self.index

    def load(self) -> VectorStoreIndex:
        """
        Load the index from disk. Initializes embedding model and node parser.
        Returns:
            VectorStoreIndex: The loaded index.
        """
        print(f"Loading index from {self.storage_dir}...")
        print("DEBUG: Calling StorageContext.from_defaults...")
        initialize_hf_embedding_model(model_name=self.embedding_model_name)
        Settings.node_parser = self.node_parser
        storage_context = StorageContext.from_defaults(
            persist_dir=self.storage_dir
        )
        self.index = load_index_from_storage(storage_context)
        print("Index loaded successfully.")
        return self.index

    def persist(self):
        """
        Persist the current index to disk.
        """
        if not self.index:
            raise RuntimeError("No index to persist. Build or load the index first.")
        print(f"Persisting index to {self.storage_dir}...")
        self.index.storage_context.persist(
            persist_dir=self.storage_dir
        )
        print("Index persisted successfully.")

    def get_index(self) -> VectorStoreIndex:
        """
        Return the underlying VectorStoreIndex object.
        Returns:
            VectorStoreIndex: The index object.
        """
        if not self.index:
            raise RuntimeError("Index not built or loaded yet.")
        return self.index 

if __name__ == "__main__": #script testing
    # python -m src.index_builder
    from src.config_loader import AppConfig
    
    app_config = AppConfig()
    indexing_config: IndexBuilderConfig = app_config.get_indexing_config()
    
    builder = IndexBuilder(config=indexing_config)
    
    time_start = time.time()
    try:
        index = builder.build(force_rebuild=True)
        print(f"\nIndex built and persisted successfully. Number of documents in index: {len(index.docstore.docs)}")
    except Exception as e:
        print(f"Index build failed: {e}") 
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")