import os
import yaml
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# Load environment variables from .env file
load_dotenv()

@dataclass
class IndexBuilderConfig:
    """Configuration settings for the IndexBuilder."""
    storage_dir: str
    embedding_model_name: str
    corpus_path: str
    corpus_id_field: str
    corpus_text_fields: List[str]
    corpus_metadata_fields: List[str]
    chunk_size: int
    chunk_overlap: int
    docstore_filename: str = "docstore.json"
    vector_store_filename: str = "vector_store.json"
    index_store_filename: str = "index_store.json"

@dataclass
class QueryEngineBuilderConfig:
    """Configuration settings for the QueryEngineBuilder."""
    embedding_model_name: str
    chunk_size: int
    chunk_overlap: int
    similarity_top_k: int = 3

@dataclass
class RetrieverConfig:
    """Configuration for the retriever component."""
    similarity_top_k: int = 3

class AppConfig:
    """
    Application configuration class.
    Loads settings from config.yaml and environment variables.
    Centralizes all project-wide configuration variables.
    Raises errors if required config values are missing.
    """
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        raw_yaml_config: Dict[str, Any] = self._load_config_yaml()

        # API Keys from .env
        self.openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")

        # --- Top Level General Configs (if any) ---
        # Example: self.some_general_setting = raw_yaml_config.get("some_general_setting", "default_value")

        # --- Nested Config Sections ---
        self.index_builder_settings: Dict[str, Any] = raw_yaml_config.get("index_builder", {})
        self.query_engine_builder_settings: Dict[str, Any] = raw_yaml_config.get("query_engine_builder", {})
        self.retriever_settings: Dict[str, Any] = raw_yaml_config.get("retriever", {})
        self.llm_settings: Dict[str, Any] = raw_yaml_config.get("llm", {})

        # Validate essential LLM configurations if LLM section exists
        if self.llm_settings:
            self.model_name: Optional[str] = self.llm_settings.get("model_name")
            self.openrouter_base_url: Optional[str] = self.llm_settings.get("openrouter_base_url")
            if not self.model_name:
                raise ValueError(f"'model_name' is missing from 'llm' section in {self.config_path}.")
            if not self.openrouter_api_key and self.openrouter_base_url: # Only warn if base_url is present but no key
                print(f"Warning: OPENROUTER_API_KEY not found in .env file. LLM functionalities might be limited if using OpenRouter with {self.openrouter_base_url}.")
        else:
            self.model_name = None
            self.openrouter_base_url = None
            print(f"Warning: 'llm' section not found in {self.config_path}. LLM functionalities will be unavailable.")

    def _load_config_yaml(self) -> Dict[str, Any]:
        """Loads configuration from the YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                if not isinstance(config_data, dict):
                    raise ValueError(f"Root of {self.config_path} must be a dictionary. Found: {type(config_data)}")
                return config_data
        except FileNotFoundError:
            raise FileNotFoundError(f"{self.config_path} not found. Please create it with all required configuration values.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing {self.config_path}: {e}.")

    def _require_from_section(self, section: Dict[str, Any], key: str, section_name: str) -> Any:
        """Get a required config value from a section or raise an error if missing."""
        value = section.get(key)
        if value is None:
            raise ValueError(f"Required config value '{key}' is missing from '{section_name}' section in {self.config_path}.")
        return value

    def _optional_from_section(self, section: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Return the config value for `key` from a section if present, else `default`."""
        return section.get(key, default)

    def get_llm_config(self) -> Optional[Dict[str, Any]]:
        """Returns a dictionary with LLM related configurations if LLM is configured."""
        if not self.model_name or not self.openrouter_api_key:
            # Allow proceeding without API key if it's not strictly for OpenRouter or model is local
            if self.model_name and not self.openrouter_base_url: # e.g. a local HF model
                 return { "model": self.model_name, "api_key": None, "base_url": None }
            print("Warning: LLM not fully configured (model_name or OPENROUTER_API_KEY missing). Returning None for LLM config.")
            return None
        return {
            "model": self.model_name,
            "api_key": self.openrouter_api_key,
            "base_url": self.openrouter_base_url
        }

    def get_index_builder_config(self) -> IndexBuilderConfig:
        """Returns an IndexBuilderConfig object."""
        cfg = self.index_builder_settings
        section_name = "index_builder"
        return IndexBuilderConfig(
            storage_dir=self._require_from_section(cfg, "storage_dir", section_name),
            embedding_model_name=self._require_from_section(cfg, "embedding_model_name", section_name),
            corpus_path=self._require_from_section(cfg, "corpus_path", section_name),
            corpus_id_field=self._require_from_section(cfg, "corpus_id_field", section_name),
            corpus_text_fields=self._require_from_section(cfg, "corpus_text_fields", section_name),
            corpus_metadata_fields=self._require_from_section(cfg, "corpus_metadata_fields", section_name),
            chunk_size=int(self._require_from_section(cfg, "chunk_size", section_name)),
            chunk_overlap=int(self._require_from_section(cfg, "chunk_overlap", section_name)),
            docstore_filename=self._optional_from_section(cfg, "docstore_filename", "docstore.json"),
            vector_store_filename=self._optional_from_section(cfg, "vector_store_filename", "vector_store.json"),
            index_store_filename=self._optional_from_section(cfg, "index_store_filename", "index_store.json")
        )

    def get_query_engine_builder_config(self) -> QueryEngineBuilderConfig:
        """Returns a QueryEngineBuilderConfig object."""
        cfg = self.query_engine_builder_settings
        section_name = "query_engine_builder"
        # These should ideally align with or be derived from the index settings if not explicitly different
        # For now, we require them in the query_engine_builder section for explicitness in the config.
        return QueryEngineBuilderConfig(
            embedding_model_name=self._require_from_section(cfg, "embedding_model_name", section_name),
            chunk_size=int(self._require_from_section(cfg, "chunk_size", section_name)),
            chunk_overlap=int(self._require_from_section(cfg, "chunk_overlap", section_name)),
            similarity_top_k=int(self._require_from_section(cfg, "similarity_top_k", section_name))
        )
    
    def get_retriever_config(self) -> RetrieverConfig:
        """Returns a RetrieverConfig object."""
        cfg = self.retriever_settings
        section_name = "retriever"
        return RetrieverConfig(
            similarity_top_k=int(self._require_from_section(cfg, "similarity_top_k", section_name))
        )

if __name__ == "__main__": #script testing
    # Create a dummy config.yaml for testing
    dummy_config_content = {
        "index_builder": {
            "storage_dir": "./test_storage",
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "corpus_path": "./dummy_corpus.json",
            "corpus_id_field": "id",
            "corpus_text_fields": ["title", "text"],
            "corpus_metadata_fields": ["meta"],
            "chunk_size": 100,
            "chunk_overlap": 10,
        },
        "query_engine_builder": {
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size": 100,
            "chunk_overlap": 10,
            "similarity_top_k": 3
        },
        "retriever": {
            "similarity_top_k": 2
        },
        "llm": {
            "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
            "openrouter_base_url": "https://openrouter.ai/api/v1"
        }
    }
    with open("temp_test_config.yaml", 'w') as f:
        yaml.dump(dummy_config_content, f)
    
    print("Testing AppConfig with temp_test_config.yaml:")
    try:
        config = AppConfig(config_path="temp_test_config.yaml")
        print("\nIndex Builder Config:")
        print(config.get_index_builder_config())
        print("\nQuery Engine Builder Config:")
        print(config.get_query_engine_builder_config())
        print("\nRetriever Config:")
        print(config.get_retriever_config())
        print("\nLLM Config:")
        print(config.get_llm_config())
        print("\nAppConfig loaded successfully.")
    except Exception as e:
        print(f"Error loading AppConfig: {e}")
    finally:
        if os.path.exists("temp_test_config.yaml"):
            os.remove("temp_test_config.yaml")