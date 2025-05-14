import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from .settings import settings # Import the Pydantic settings

# Define the path to the config file
CONFIG_FILE_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

def load_config() -> Dict[str, Any]:
    """Loads the configuration from the YAML file."""
    if not CONFIG_FILE_PATH.is_file():
        raise FileNotFoundError(
            f"Configuration file not found at {CONFIG_FILE_PATH}. "
            "Please ensure 'config.yaml' exists in the root directory."
        )
    with open(CONFIG_FILE_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config

# Load the configuration when this module is imported
_config_data = load_config()

# Provide easy access to specific configuration values from config.yaml
MODEL_NAME: Optional[str] = _config_data.get("model_name")
OPENROUTER_BASE_URL: Optional[str] = _config_data.get("OPENROUTER_BASE_URL")

# API Keys from settings.py (environment variables)
OPENROUTER_API_KEY = settings.OPENROUTER_API_KEY

# Derived paths (relative to repo root)
DATA_DIR = Path("data")
VECTOR_DIR = Path("vectorspace")
INDEX_PATH = VECTOR_DIR / "naacl2025.index"

def get_config_value(key: str, default: Optional[Any] = None) -> Optional[Any]:
    """Helper function to get a configuration value by key."""
    return _config_data.get(key, default)

# Example of how to use it in other modules:
# from llm_agent.config import MODEL_NAME, get_config_value, OPENROUTER_BASE_URL
# print(f"Using model: {MODEL_NAME}")
# print(f"OpenRouter URL: {OPENROUTER_BASE_URL}")
# custom_setting = get_config_value("custom_setting", "default_value")
# print(f"Custom setting: {custom_setting}")

if __name__ == "__main__":
    # For testing purposes, print the loaded config
    print("Loaded configuration from config.yaml:")
    for key, value in _config_data.items():
        print(f"  {key}: {value}")
    print(f"Model Name (from YAML): {MODEL_NAME}")
    print(f"OpenRouter Base URL (from YAML): {OPENROUTER_BASE_URL}")
    print(f"OpenRouter API Key (from .env via settings): {OPENROUTER_API_KEY}") 