import json
from typing import List, Dict, Any
from llama_index.core import Document

class AnthologyLoader:
    """
    Loads ACL Anthology data from a JSON file and converts entries to LlamaIndex Documents.
    """
    def __init__(self, json_file_path: str = "data/anthology+abstracts.json"):
        self.json_file_path = json_file_path

    def load_data(self) -> List[Document]:
        """
        Reads the JSON file and converts each paper entry into a LlamaIndex Document.
        
        The main text of the Document is formed from the title and abstract.
        Other fields like authors, year, URL are stored in metadata.
        """
        documents: List[Document] = []
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {self.json_file_path}. Please ensure it exists.")
            print("You might need to run the bib_to_json conversion first.")
            return documents # Return empty list
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {self.json_file_path}: {e}")
            return documents # Return empty list

        if not isinstance(data, list):
            print(f"Error: Expected a list of entries in {self.json_file_path}, but got {type(data)}.")
            return documents

        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                print(f"Warning: Skipping entry #{i} as it is not a dictionary: {entry}")
                continue

            # Extract common fields, providing defaults for robustness
            title = entry.get("title", "No Title Provided")
            abstract = entry.get("abstract", "No Abstract Provided")
            authors_list = entry.get("author", []) # Assuming author is a list of strings or list of dicts
            year = entry.get("year", "Unknown Year")
            url = entry.get("url", "No URL Provided")
            # Assuming 'id' or a unique key might be present, which is good for `doc_id`
            doc_id = entry.get("id", entry.get("ID", f"entry_{i}")) # Use 'ID' if 'id' not present, else generate

            # Format authors into a string if it's a list of strings
            if authors_list and isinstance(authors_list, list) and all(isinstance(a, str) for a in authors_list):
                authors_str = ", ".join(authors_list)
            elif authors_list and isinstance(authors_list, list) and all(isinstance(a, dict) and "name" in a for a in authors_list):
                 authors_str = ", ".join([a.get("name", "Unknown Author") for a in authors_list])
            elif isinstance(authors_list, str): # if author is just a string
                authors_str = authors_list
            else:
                authors_str = "Unknown Authors"
            
            # Construct the main text for the Document
            # We can refine what goes into the main text vs. metadata
            text_content = f"Title: {title}\nAbstract: {abstract}"
            
            # Prepare metadata
            metadata = {
                "title": title,
                "authors": authors_str,
                "year": str(year), # Ensure year is a string for metadata
                "url": url,
                "source_file": self.json_file_path
            }
            
            # Create LlamaIndex Document
            # We use a specific doc_id for better tracking and potential updates later
            doc = Document(
                text=text_content,
                metadata=metadata,
                doc_id=str(doc_id) # doc_id should be a string
            )
            documents.append(doc)
            
        if not documents:
            print(f"No documents were loaded from {self.json_file_path}. Check the file content and format.")
        else:
            print(f"Successfully loaded {len(documents)} documents from {self.json_file_path}.")

        return documents

# Example usage (optional, for direct script testing)
if __name__ == "__main__":
    loader = AnthologyLoader()
    anthology_docs = loader.load_data()
    if anthology_docs:
        print(f"\nFirst document loaded ({anthology_docs[0].doc_id}):")
        print(f"Text: {anthology_docs[0].text[:200]}...")
        print(f"Metadata: {anthology_docs[0].metadata}") 