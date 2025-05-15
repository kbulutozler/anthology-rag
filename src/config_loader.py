import os
import yaml
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables from .env file
load_dotenv()

DEFAULT_MODEL_NAME = "google/gemini-flash-1.5"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

class AppConfig:
    """
    Application configuration class.
    Loads settings from config.yaml and environment variables.
    """
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.settings: Dict[str, Any] = self._load_config_yaml()

        # API Keys from .env
        self.openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")

        # Model and API settings from config.yaml, with fallbacks
        self.model_name: str = self.settings.get("model_name", DEFAULT_MODEL_NAME)
        self.openrouter_base_url: str = self.settings.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)

        # Validate essential configurations
        if not self.openrouter_api_key:
            print("Warning: OPENROUTER_API_KEY not found in .env file. LLM functionalities will be limited.")
        
        # Potentially add more validations here

    def _load_config_yaml(self) -> Dict[str, Any]:
        """Loads configuration from the YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                if config_data is None: # Handle empty YAML file
                    return {}
                return config_data
        except FileNotFoundError:
            print(f"Warning: {self.config_path} not found. Using default settings.")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing {self.config_path}: {e}. Using default settings.")
            return {}

    def get_llm_config(self) -> Dict[str, Any]:
        """Returns a dictionary with LLM related configurations."""
        return {
            "model": self.model_name,
            "api_key": self.openrouter_api_key,
            "base_url": self.openrouter_base_url
        }

# Example usage (optional, for direct script testing)
if __name__ == "__main__":
    config = AppConfig()
    print(f"Model Name: {config.model_name}")
    print(f"OpenRouter API Key Loaded: {'Yes' if config.openrouter_api_key else 'No'}")
    print(f"OpenRouter Base URL: {config.openrouter_base_url}")
    llm_params = config.get_llm_config()
    print(f"LLM Config for LlamaIndex: {llm_params}") 