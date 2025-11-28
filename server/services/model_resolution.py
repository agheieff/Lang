"""
Model resolution for LLM tasks.

This module provides the interface between the generation/translation services
and the new UserModelConfig-based model management system.
"""
from typing import List, Optional, Literal
from sqlalchemy.orm import Session
from server.models import UserModelConfig, Profile
from server.auth import Account
from server.llm_config.llm_models import ModelConfig
from server.services.user_model_service import get_user_model_service, TaskType

# Map preference keys to task types
PREFERENCE_TO_TASK: dict[str, TaskType] = {
    "preferred_generation_model": "generation",
    "preferred_translation_model": "word_translation",  # Legacy mapping
    "preferred_word_translation_model": "word_translation",
    "preferred_sentence_translation_model": "sentence_translation",
}


def resolve_models_for_task(
    account_db: Session,
    global_db: Session,
    account_id: int,
    lang: Optional[str] = None,
    preference_key: Optional[str] = None
) -> List[ModelConfig]:
    """
    Resolve the list of models to try for a task, respecting user preferences and providers.
    
    This function bridges the old interface with the new UserModelConfig system.
    
    Args:
        account_db: User's private DB session
        global_db: Global DB session (for tier)
        account_id: User ID
        lang: Current language context (for profile)
        preference_key: Key in profile.settings (legacy) or task type indicator
        
    Returns:
        List of ModelConfig objects in priority order.
    """
    service = get_user_model_service()
    
    # Get user's subscription tier
    account = global_db.query(Account).filter(Account.id == account_id).first()
    user_tier = account.subscription_tier if account else "Free"
    
    # Ensure user has models (injects system defaults if needed)
    service.ensure_user_has_models(account_db, account_id, tier=user_tier)
    
    # Determine task type from preference key
    task_type: TaskType = PREFERENCE_TO_TASK.get(preference_key, "generation")
    
    # Get the assigned model for this task
    assigned = service.get_assigned_model(account_db, account_id, task_type)
    
    # Get all active models as fallbacks
    all_models = service.list_models(account_db, account_id, include_inactive=False)
    
    models_to_try: List[ModelConfig] = []
    seen_ids: set[int] = set()
    
    def add_model(config: UserModelConfig):
        if config.id in seen_ids:
            return
        seen_ids.add(config.id)
        
        # Convert UserModelConfig to ModelConfig for compatibility
        mc = _user_model_to_model_config(config)
        models_to_try.append(mc)
    
    # 1. Add assigned model first
    if assigned:
        add_model(assigned)
    
    # 2. Add remaining models by priority
    for config in all_models:
        add_model(config)
    
    # 3. If still empty, get system fallback
    if not models_to_try:
        resolved = service._get_system_fallback()
        mc = ModelConfig(
            id="system-fallback",
            display_name="System Default",
            model=resolved.model_id,
            base_url=resolved.base_url,
            api_key_env=None,
            _api_key=resolved.api_key,
            max_tokens=resolved.max_tokens or 32768,
            allowed_tiers=["Free", "Standard", "Pro", "Pro+", "BYOK", "admin"],
            capabilities=["text"]
        )
        models_to_try.append(mc)
    
    return models_to_try


def _user_model_to_model_config(config: UserModelConfig) -> ModelConfig:
    """Convert a UserModelConfig to the legacy ModelConfig format."""
    import os
    
    # Determine API key
    api_key = config.api_key
    if not api_key and config.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
    
    # Determine base URL
    base_url = config.base_url
    if not base_url:
        if config.provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        elif config.provider == "openai":
            base_url = "https://api.openai.com/v1"
        elif config.provider == "anthropic":
            base_url = "https://api.anthropic.com/v1"
        elif config.provider == "local":
            base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
        else:
            base_url = "https://openrouter.ai/api/v1"
    
    return ModelConfig(
        id=f"user-{config.id}",
        display_name=config.display_name,
        model=config.model_id,
        base_url=base_url,
        api_key_env=None,
        _api_key=api_key,
        max_tokens=config.max_tokens or 32768,
        allowed_tiers=["Free", "Standard", "Pro", "Pro+", "BYOK", "admin"],
        capabilities=config.capabilities or ["text"]
    )


def resolve_model_for_translation(
    account_db: Session,
    global_db: Session,
    account_id: int,
    translation_type: Literal["word", "sentence"] = "word"
) -> List[ModelConfig]:
    """
    Convenience function for translation tasks.
    
    Args:
        account_db: User's private DB session
        global_db: Global DB session
        account_id: User ID
        translation_type: "word" or "sentence"
        
    Returns:
        List of ModelConfig objects in priority order.
    """
    preference_key = f"preferred_{translation_type}_translation_model"
    return resolve_models_for_task(account_db, global_db, account_id, preference_key=preference_key)
