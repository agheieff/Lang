from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    id: str
    display_name: str
    model: str
    base_url: str
    api_key_env: Optional[str]
    max_tokens: int
    allowed_tiers: List[str]
    capabilities: List[str] = field(default_factory=list)
    cost_per_token: float = 0.0
    
    def get_api_key(self) -> Optional[str]:
        if not self.api_key_env:
            return None
        return os.getenv(self.api_key_env)


@dataclass
class ProviderConfig:
    max_retries: int = 2
    timeout: int = 60
    referer: Optional[str] = None
    app_title: Optional[str] = None


@dataclass
class LLMModelsConfig:
    models: List[ModelConfig] = field(default_factory=list)
    default_model: str = "kimi-k2-0905"
    provider_configs: Dict[str, ProviderConfig] = field(default_factory=dict)
    fallback_chain: List[str] = field(default_factory=list)
    
    def get_model_by_id(self, model_id: str) -> Optional[ModelConfig]:
        for model in self.models:
            if model.id == model_id:
                return model
        return None
    
    def get_models_for_tier(self, tier: str) -> List[ModelConfig]:
        return [model for model in self.models if tier in model.allowed_tiers]
    
    def get_provider_config(self, provider: str) -> ProviderConfig:
        return self.provider_configs.get(provider, ProviderConfig())


class LLMConfigLoader:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                "llm_models.json"
            )
        self.config_path = Path(config_path)
        self._config: Optional[LLMModelsConfig] = None
    
    def _load_from_env(self) -> LLMModelsConfig:
        """Fallback to environment variables for backward compatibility"""
        config = LLMModelsConfig()
        
        # Create a basic model from environment variables
        if os.getenv("OPENROUTER_API_KEY"):
            openrouter_model = ModelConfig(
                id="env-openrouter",
                display_name="OpenRouter (Env)",
                model=os.getenv("OPENROUTER_MODEL", "moonshotai/kimi-k2:free"),
                base_url="https://openrouter.ai/api/v1",
                api_key_env="OPENROUTER_API_KEY",
                max_tokens=4096,
                allowed_tiers=["Free", "Standard", "Pro", "Pro+", "BYOK", "admin"],
                capabilities=["text"]
            )
            config.models.append(openrouter_model)
            config.default_model = "env-openrouter"
            
            # Provider config
            config.provider_configs["openrouter"] = ProviderConfig(
                max_retries=int(os.getenv("OPENROUTER_MAX_RETRIES", "2")),
                timeout=60,
                app_title=os.getenv("OPENROUTER_APP_TITLE", "Arcadia AI Chat"),
                referer=os.getenv("OPENROUTER_REFERER")
            )
        
        # Local model config
        local_base = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
        local_model = ModelConfig(
            id="env-local",
            display_name="Local Model (Env)",
            model="local",
            base_url=local_base,
            api_key_env=None,
            max_tokens=16384,
            allowed_tiers=["Free", "Standard", "Pro", "Pro+", "BYOK", "admin"],
            capabilities=["text"]
        )
        config.models.append(local_model)
        
        if not config.default_model:
            config.default_model = "env-local"
            
        config.provider_configs["local"] = ProviderConfig(timeout=30)
        
        return config
    
    def _load_from_json(self) -> LLMModelsConfig:
        """Load configuration from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"LLM config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            data = json.load(f)
        
        config = LLMModelsConfig()
        
        # Parse models
        for model_data in data.get("models", []):
            model = ModelConfig(**model_data)
            config.models.append(model)
        
        # Parse other config
        config.default_model = data.get("default_model", "kimi-k2-0905")
        config.fallback_chain = data.get("fallback_chain", [])
        
        # Parse provider configs
        for provider, provider_data in data.get("provider_configs", {}).items():
            config.provider_configs[provider] = ProviderConfig(**provider_data)
        
        return config
    
    def load_config(self) -> LLMModelsConfig:
        """Load configuration from JSON or fall back to environment variables"""
        try:
            if self.config_path.exists():
                return self._load_from_json()
            else:
                print(f"Warning: LLM config file not found at {self.config_path}, using environment variables")
                return self._load_from_env()
        except Exception as e:
            print(f"Warning: Failed to load LLM config: {e}, using environment variables")
            return self._load_from_env()
    
    @property
    def config(self) -> LLMModelsConfig:
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def reload(self) -> LLMModelsConfig:
        """Force reload of configuration"""
        self._config = None
        return self.config


# Global instance
_llm_config_loader = LLMConfigLoader()

def get_llm_config() -> LLMModelsConfig:
    """Get the global LLM configuration"""
    return _llm_config_loader.config

def reload_llm_config() -> LLMModelsConfig:
    """Reload the global LLM configuration"""
    return _llm_config_loader.reload()
