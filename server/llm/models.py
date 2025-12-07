"""
LLM model configuration loading from models.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""
    id: str
    display_name: str
    model: str
    base_url: str
    api_key_env: Optional[str] = None
    max_tokens: Optional[int] = None
    allowed_tiers: Optional[List[str]] = None
    capabilities: Optional[List[str]] = None
    cost_per_token: Optional[float] = None


class LLMModelsConfig:
    """Configuration container for LLM models."""
    
    def __init__(self, data: Dict[str, Any]):
        self.models: List[ModelConfig] = []
        self.default_model: str = data.get("default_model", "grok-fast-free")
        self.fallback_chain: List[str] = data.get("fallback_chain", [])
        self.provider_configs: Dict[str, Any] = data.get("provider_configs", {})
        
        # Parse model configs
        for model_data in data.get("models", []):
            config = ModelConfig(
                id=model_data["id"],
                display_name=model_data["display_name"],
                model=model_data["model"],
                base_url=model_data["base_url"],
                api_key_env=model_data.get("api_key_env"),
                max_tokens=model_data.get("max_tokens"),
                allowed_tiers=model_data.get("allowed_tiers", []),
                capabilities=model_data.get("capabilities", []),
                cost_per_token=model_data.get("cost_per_token"),
            )
            self.models.append(config)
    
    def get_models_for_tier(self, tier: str) -> List[ModelConfig]:
        """Get all models available for a given tier."""
        return [
            model for model in self.models
            if tier in (model.allowed_tiers or [])
        ]
    
    def get_model_by_id(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration by ID."""
        for model in self.models:
            if model.id == model_id:
                return model
        return None
    
    def get_default_model_config(self) -> Optional[ModelConfig]:
        """Get the default model configuration."""
        return self.get_model_by_id(self.default_model)


def get_llm_config() -> LLMModelsConfig:
    """Load LLM configuration from models.json file."""
    models_file = Path(__file__).parent / "models.json"
    
    try:
        with open(models_file, "r") as f:
            data = json.load(f)
        return LLMModelsConfig(data)
    except Exception as e:
        # Return empty config if file not found or invalid
        return LLMModelsConfig(
            {
                "models": [],
                "default_model": "grok-fast-free",
                "fallback_chain": [],
                "provider_configs": {}
            }
        )
