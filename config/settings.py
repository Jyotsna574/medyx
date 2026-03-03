"""Configuration management for MedicalAgentDiagnosis-MAD.

Loads settings from environment variables and YAML.
Supports: Gemini (cloud) and local HuggingFace models.
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
    google_api_key: str = Field(default="", description="Google/Gemini API key")


    # Database URIs
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="", description="Neo4j password")

    # Model Selection
    active_provider: str = Field(default="", description="gemini | local")
    local_active_model: str = Field(
        default="",
        description="Override active_model for local HuggingFace provider. Options: biomistral_7b, med42_8b, meditron_70b, clinical_camel_70b, openbiollm_70b"
    )
    local_model_path: str = Field(
        default="",
        description="Path to pre-downloaded model directory. Skips HuggingFace download if set."
    )

    # Application Settings
    app_name: str = Field(default="MedicalAgentDiagnosis-MAD", description="Application name")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")


class ModelConfig:
    """Configuration for LLM models loaded from models.yaml.
    
    Handles both cloud API providers and local HuggingFace models.
    Environment variables take precedence over YAML configuration.
    """
    
    def __init__(self):
        self._config = load_yaml_config("models.yaml")
        self._settings = Settings()
        self._reload_active_provider()
    
    def _reload_active_provider(self):
        """Determine the active provider from env vars or config."""
        # Environment variable takes precedence
        if self._settings.active_provider:
            self._active_provider = self._settings.active_provider
        else:
            self._active_provider = self._config.get("active_provider", "gemini")
    
    @property
    def active_provider(self) -> str:
        """Get the currently active LLM provider name."""
        return self._active_provider
    
    @property
    def is_local_provider(self) -> bool:
        """Check if the active provider is a local HuggingFace model."""
        return self._active_provider == "local"
    
    def get_provider_config(self, provider: Optional[str] = None) -> dict[str, Any]:
        """Get configuration for a specific provider.
        
        Args:
            provider: Provider name (gemini or local). 
                     Uses active provider if None.
            
        Returns:
            Provider configuration dict with model, temperature, max_tokens, etc.
        """
        provider = provider or self._active_provider
        providers = self._config.get("providers", {})
        return providers.get(provider, {})
    
    def get_model_name(self, provider: Optional[str] = None) -> str:
        """Get the model name for a provider."""
        config = self.get_provider_config(provider)
        return config.get("model", "")
    
    def get_temperature(self, provider: Optional[str] = None) -> float:
        """Get the temperature setting for a provider."""
        config = self.get_provider_config(provider)
        
        # For local provider, check default_params
        if (provider or self._active_provider) == "local":
            default_params = config.get("default_params", {})
            return default_params.get("temperature", 0.7)
        
        return config.get("temperature", 0.7)
    
    def get_max_tokens(self, provider: Optional[str] = None) -> int:
        """Get the max_tokens setting for a provider."""
        config = self.get_provider_config(provider)
        
        # For local provider, check default_params
        if (provider or self._active_provider) == "local":
            default_params = config.get("default_params", {})
            return default_params.get("max_new_tokens", 2048)
        
        return config.get("max_tokens", 2048)
    
    def get_local_model_config(self) -> dict[str, Any]:
        """Get configuration for the active local HuggingFace model.
        
        Returns:
            Dictionary with repo_id, load_in_4bit, device_map, etc.
            Returns empty dict if local provider is not configured.
        """
        local_config = self.get_provider_config("local")
        if not local_config:
            return {}
        
        # Determine which local model to use
        # Priority: LOCAL_ACTIVE_MODEL env var > active_model in yaml
        if self._settings.local_active_model:
            active_model = self._settings.local_active_model
        else:
            active_model = local_config.get("active_model", "med42_8b")
        
        models = local_config.get("models", {})
        model_config = models.get(active_model, {})
        
        # Add active model name to the config
        model_config["model_name"] = active_model
        
        # Add default generation params
        model_config["generation_params"] = local_config.get("default_params", {})
        
        return model_config
    
    def get_local_model_path(self) -> Optional[str]:
        """Get the local model path if set via environment variable.
        
        Returns:
            Path string if LOCAL_MODEL_PATH is set, None otherwise.
        """
        if self._settings.local_model_path:
            return self._settings.local_model_path
        return None
    
    def get_api_key(self, provider: Optional[str] = None) -> str:
        """Get the API key for a cloud provider.
        
        Args:
            provider: Provider name. Uses active provider if None.
            
        Returns:
            API key string, or empty string if not found.
        """
        provider = provider or self._active_provider
        config = self.get_provider_config(provider)
        
        # Get the environment variable name for the API key
        api_key_env = config.get("api_key_env", "")
        if api_key_env:
            return os.getenv(api_key_env, "")
        
        if provider == "gemini":
            return self._settings.google_api_key
        return ""
    
    def reload(self):
        """Reload configuration from disk and environment."""
        self._config = load_yaml_config("models.yaml")
        self._settings = Settings()
        self._reload_active_provider()


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
