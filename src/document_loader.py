import json
import time
from typing import List, Dict, Any, Optional
from llama_index.core import Document

class DocumentLoader:
    """
    Loads document data from a JSON file and converts entries to LlamaIndex Documents.
    The loader is domain-agnostic and can handle any document collection.
    Optionally, a list of metadata fields can be specified for extraction.

    Args:
        data_path (str): Path to the input JSON data file. Must be provided explicitly.
        metadata_fields (Optional[List[str]]): List of metadata fields to extract (optional).
    """
    def __init__(self, corpus_path: str, text_fields: List[str], metadata_fields: List[str], id_field: str):
        self.corpus_path = corpus_path
        self.text_fields = text_fields
        self.metadata_fields = metadata_fields
        self.id_field = id_field

    def load_data(self) -> List[Document]:
        """
        Reads the corpus JSON file and converts each entry into a LlamaIndex Document.
        text is formed from the specified text fields, metadata is formed from the specified metadata fields.
        """
        documents: List[Document] = []
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {self.corpus_path}. Please ensure it exists.")
            return documents
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {self.corpus_path}: {e}")
            return documents

        if not isinstance(data, list):
            print(f"Error: Expected a list of entries in {self.corpus_path}, but got {type(data)}.")
            return documents

        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                print(f"Warning: Skipping entry #{i} as it is not a dictionary: {entry}")
                continue

            # Extract main text
            text_content = ""
            for field in self.text_fields:
                text_content += f"{field}: {entry.get(field, "")}\n"
                

            # Prepare metadata
            metadata = {}
            for field in self.metadata_fields:
                metadata[field] = entry.get(field, None)

            # if entry has no id field, give a unique id
            doc_id = entry.get(self.id_field, f"entry_{i}")

            doc = Document(
                text=text_content,
                metadata=metadata,
                doc_id=str(doc_id)
            )
            documents.append(doc)

        if not documents:
            print(f"No documents were loaded from {self.corpus_path}. Check the file content and format.")
        else:
            print(f"Successfully loaded {len(documents)} documents from {self.corpus_path}.")

        return documents

if __name__ == "__main__": #script testing
    # python -m src.document_loader
    from src.config_loader import AppConfig
    config = AppConfig()
    loader = DocumentLoader(config.corpus_path, config.corpus_text_fields, config.corpus_metadata_fields, config.corpus_id_field)
    time_start = time.time()
    try:
        documents = loader.load_data()
        time_end = time.time()
        print(f"\nTotal documents loaded: {len(documents)} in {time_end - time_start:.2f} seconds")
    except Exception as e:
        print(f"Error loading documents: {e}")
        exit(1)
    