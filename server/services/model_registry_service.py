from __future__ import annotations

import logging
from typing import Optional, List, Dict, Set
from server.llm.models import get_llm_config, ModelConfig, LLMModelsConfig

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when a requested model is not found"""
    pass


class ModelAccessDeniedError(Exception):
    """Raised when user tier doesn't have access to requested model"""
    pass


class ModelRegistryService:
    """Central service for managing LLM model selection and validation"""
    
    def __init__(self):
        self.config: LLMModelsConfig = get_llm_config()
    
    def get_available_models(self, tier: str) -> List[ModelConfig]:
        """Get all models available for a specific user tier"""
        return self.config.get_models_for_tier(tier)
    
    def get_model_by_id(self, model_id: str, require_tier: Optional[str] = None) -> ModelConfig:
        """
        Get model by ID, optionally checking tier access
        
        Args:
            model_id: The model ID to look up
            require_tier: If provided, check that this tier has access
            
        Returns:
            ModelConfig: The model configuration
            
        Raises:
            ModelNotFoundError: If model is not found
            ModelAccessDeniedError: If tier access is required but denied
        """
        model = self.config.get_model_by_id(model_id)
        if not model:
            raise ModelNotFoundError(f"Model '{model_id}' not found")
        
        if require_tier and require_tier not in model.allowed_tiers:
            raise ModelAccessDeniedError(
                f"Tier '{require_tier}' does not have access to model '{model_id}'. "
                f"Allowed tiers: {', '.join(model.allowed_tiers)}"
            )
        
        return model
    
    def get_default_model(self, tier: str) -> ModelConfig:
        """
        Get the default model for a user tier
        
        Args:
            tier: The user's subscription tier
            
        Returns:
            ModelConfig: The default model for this tier
        """
        # First try the configured default model
        default_model = self.config.get_model_by_id(self.config.default_model)
        if default_model and tier in default_model.allowed_tiers:
            return default_model
        
        # Fall back to any available model
        available_models = self.get_available_models(tier)
        if not available_models:
            raise ModelAccessDeniedError(f"No models available for tier '{tier}'")
        
        # Prefer models in the fallback chain
        for model_id in self.config.fallback_chain:
            for model in available_models:
                if model.id == model_id:
                    return model
        
        # Return first available model as last resort
        return available_models[0]
    
    def validate_model_access(self, model_id: str, tier: str) -> bool:
        """
        Check if a tier has access to a specific model
        
        Args:
            model_id: The model ID to check
            tier: The user's subscription tier
            
        Returns:
            bool: True if access is granted
        """
        try:
            self.get_model_by_id(model_id, require_tier=tier)
            return True
        except (ModelNotFoundError, ModelAccessDeniedError):
            return False
    
    def get_provider_config(self, provider: str):
        """Get provider-specific configuration"""
        return self.config.get_provider_config(provider)
    
    def get_models_by_capability(self, capability: str, tier: str) -> List[ModelConfig]:
        """
        Get all models with a specific capability available to a tier
        
        Args:
            capability: The capability to filter by (e.g., "text", "code", "reasoning")
            tier: The user's subscription tier
            
        Returns:
            List[ModelConfig]: Models with the requested capability
        """
        models = self.get_available_models(tier)
        return [model for model in models if capability in model.capabilities]
    
    def get_most_cost_effective_model(self, capabilities: List[str], tier: str) -> ModelConfig:
        """
        Get the most cost-effective model that supports the required capabilities
        
        Args:
            capabilities: List of required capabilities
            tier: The user's subscription tier
            
        Returns:
            ModelConfig: The most cost-effective model
        """
        # Filter models that support all required capabilities
        models = self.get_available_models(tier)
        suitable_models = []
        
        for model in models:
            if all(capability in model.capabilities for capability in capabilities):
                suitable_models.append(model)
        
        if not suitable_models:
            raise ModelNotFoundError(f"No models found supporting capabilities: {capabilities}")
        
        # Sort by cost and return the cheapest
        return sorted(suitable_models, key=lambda m: m.cost_per_token)[0]
    
    def reload_config(self) -> None:
        """Reload the configuration from disk"""
        from server.llm_config.llm_models import reload_llm_config
        self.config = reload_llm_config()
        logger.info("ModelRegistry configuration reloaded")


# Singleton instance
_model_registry = ModelRegistryService()


def get_model_registry() -> ModelRegistryService:
    """Get the global model registry instance"""
    return _model_registry
