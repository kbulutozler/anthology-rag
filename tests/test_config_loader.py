import os
import unittest
import yaml
from unittest.mock import patch, mock_open

# Ensure src directory is in Python path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from config_loader import AppConfig, DEFAULT_MODEL_NAME, DEFAULT_OPENROUTER_BASE_URL

class TestConfigLoader(unittest.TestCase):

    def setUp(self):
        # Create a dummy .env for testing if it doesn't exist or override
        self.test_api_key = "test_openrouter_api_key_123"
        # For tests, we directly manipulate environment variables or mock os.getenv

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "env_api_key_test"})
    def test_load_from_env_only(self):
        """Test loading OpenRouter API key from environment variable when config.yaml is missing."""
        # Mock open to simulate config.yaml not found
        with patch("builtins.open", mock_open()) as mocked_file:
            mocked_file.side_effect = FileNotFoundError
            config = AppConfig(config_path="dummy_config_that_does_not_exist.yaml")
            
            self.assertEqual(config.openrouter_api_key, "env_api_key_test")
            self.assertEqual(config.model_name, DEFAULT_MODEL_NAME) # Should use default
            self.assertEqual(config.openrouter_base_url, DEFAULT_OPENROUTER_BASE_URL) # Should use default

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "env_api_key_for_yaml_test"})
    def test_load_from_yaml_and_env(self):
        """Test loading settings from a dummy config.yaml and API key from .env."""
        dummy_config_content = {
            "model_name": "custom/model-from-yaml",
            "OPENROUTER_BASE_URL": "https://custom.api.com/v1"
        }
        mock_yaml_content = yaml.dump(dummy_config_content)
        
        with patch("builtins.open", mock_open(read_data=mock_yaml_content)) as mocked_file:
            config = AppConfig(config_path="dummy_config.yaml")
            mocked_file.assert_called_once_with("dummy_config.yaml", 'r')
            
            self.assertEqual(config.openrouter_api_key, "env_api_key_for_yaml_test")
            self.assertEqual(config.model_name, "custom/model-from-yaml")
            self.assertEqual(config.openrouter_base_url, "https://custom.api.com/v1")
            llm_conf = config.get_llm_config()
            self.assertEqual(llm_conf["model"], "custom/model-from-yaml")
            self.assertEqual(llm_conf["api_key"], "env_api_key_for_yaml_test")
            self.assertEqual(llm_conf["base_url"], "https://custom.api.com/v1")

    def test_load_with_missing_env_variable(self):
        """Test behavior when OPENROUTER_API_KEY is not set in .env."""
        # Ensure OPENROUTER_API_KEY is not in the environment for this test
        with patch.dict(os.environ, {}, clear=True):
             # Simulate config.yaml not found to isolate env var check
            with patch("builtins.open", mock_open()) as mocked_file:
                mocked_file.side_effect = FileNotFoundError
                with patch('builtins.print') as mock_print: # Suppress warning print
                    config = AppConfig(config_path="non_existent.yaml")
                    self.assertIsNone(config.openrouter_api_key)
                    # Check if the warning was printed (optional, but good for verifying behavior)
                    # mock_print.assert_any_call("Warning: OPENROUTER_API_KEY not found in .env file. LLM functionalities will be limited.")

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "some_key"})
    def test_load_with_empty_yaml_file(self):
        """Test loading with an empty (but valid) YAML file."""
        with patch("builtins.open", mock_open(read_data="")) as mocked_file:
            config = AppConfig(config_path="empty_config.yaml")
            mocked_file.assert_called_once_with("empty_config.yaml", 'r')
            self.assertEqual(config.openrouter_api_key, "some_key")
            self.assertEqual(config.model_name, DEFAULT_MODEL_NAME) # Fallback to default
            self.assertEqual(config.openrouter_base_url, DEFAULT_OPENROUTER_BASE_URL) # Fallback to default

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "another_key"})
    def test_load_with_malformed_yaml_file(self):
        """Test loading with a malformed YAML file."""
        malformed_yaml_content = "model_name: [unclosed bracket"
        with patch("builtins.open", mock_open(read_data=malformed_yaml_content)) as mocked_file:
            with patch('builtins.print') as mock_print: # Suppress error print for cleaner test output
                config = AppConfig(config_path="malformed_config.yaml")
                mocked_file.assert_called_once_with("malformed_config.yaml", 'r')
                self.assertEqual(config.openrouter_api_key, "another_key")
                self.assertEqual(config.model_name, DEFAULT_MODEL_NAME) # Fallback
                self.assertEqual(config.openrouter_base_url, DEFAULT_OPENROUTER_BASE_URL) # Fallback
                # mock_print.assert_any_call(f"Error parsing malformed_config.yaml: ... Using default settings.")

if __name__ == "__main__":
    unittest.main() 