# settings.py - Environment-based configuration
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Manages application settings using Pydantic.

    Attributes:
        OPENROUTER_API_KEY: The API key for OpenRouter.
        # OPENROUTER_BASE_URL is now in config.yaml
    """
    OPENROUTER_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings() 