"""Configuration management for MedicalAgentDiagnosis-MAD.

Loads settings from environment variables and YAML configuration files.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root and config directory
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


def load_yaml_config(filename: str) -> dict[str, Any]:
    """Load a YAML configuration file from the config directory.
    
    Args:
        filename: Name of the YAML file (e.g., 'models.yaml')
        
    Returns:
        Parsed YAML content as dictionary, or empty dict if file not found.
    """
    config_path = CONFIG_DIR / filename
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    openai_api_key: str = Field(default="", description="OpenAI API key")
    google_api_key: str = Field(default="", description="Google/Gemini API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    # Database URIs
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="", description="Neo4j password")

    # Application Settings
    app_name: str = Field(default="MedicalAgentDiagnosis-MAD", description="Application name")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")


class ModelConfig:
    """Configuration for LLM models loaded from models.yaml."""
    
    def __init__(self):
        self._config = load_yaml_config("models.yaml")
        self._default = self._config.get("default_provider", "gemini")
    
    @property
    def default_provider(self) -> str:
        """Get the default LLM provider name."""
        return self._default
    
    def get_provider_config(self, provider: Optional[str] = None) -> dict[str, Any]:
        """Get configuration for a specific provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'gemini'). Uses default if None.
            
        Returns:
            Provider configuration dict with model, temperature, max_tokens, etc.
        """
        provider = provider or self._default
        providers = self._config.get("providers", {})
        return providers.get(provider, {})
    
    def get_model_name(self, provider: Optional[str] = None) -> str:
        """Get the model name for a provider."""
        config = self.get_provider_config(provider)
        return config.get("model", "")
    
    def get_temperature(self, provider: Optional[str] = None) -> float:
        """Get the temperature setting for a provider."""
        config = self.get_provider_config(provider)
        return config.get("temperature", 0.7)
    
    def get_max_tokens(self, provider: Optional[str] = None) -> int:
        """Get the max_tokens setting for a provider."""
        config = self.get_provider_config(provider)
        return config.get("max_tokens", 2048)
    
    def reload(self):
        """Reload configuration from disk."""
        self._config = load_yaml_config("models.yaml")
        self._default = self._config.get("default_provider", "gemini")


class ExpertConfig:
    """Configuration for expert agents loaded from experts.yaml."""
    
    def __init__(self):
        self._config = load_yaml_config("experts.yaml")
    
    def get_expert(self, role_id: str) -> dict[str, Any]:
        """Get configuration for a specific expert role.
        
        Args:
            role_id: Expert role identifier (e.g., 'radiologist', 'pulmonologist')
            
        Returns:
            Expert configuration with system_prompt, constraints, etc.
        """
        experts = self._config.get("experts", {})
        return experts.get(role_id, {})
    
    def get_system_prompt(self, role_id: str) -> str:
        """Get the system prompt for an expert role."""
        expert = self.get_expert(role_id)
        return expert.get("system_prompt", "")
    
    def get_all_expert_ids(self) -> list[str]:
        """Get all configured expert role IDs."""
        return list(self._config.get("experts", {}).keys())
    
    def get_consultation_workflow(self) -> list[dict[str, Any]]:
        """Get the consultation workflow configuration."""
        return self._config.get("workflow", [])
    
    def reload(self):
        """Reload configuration from disk."""
        self._config = load_yaml_config("experts.yaml")


# Singleton instances
settings = Settings()
model_config = ModelConfig()
expert_config = ExpertConfig()
