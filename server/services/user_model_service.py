"""
User Model Configuration Service.

Handles CRUD operations for user's LLM model configurations and
model resolution for different tasks (generation, translation).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Literal

from sqlalchemy.orm import Session

from ..models import UserModelConfig
from ..llm.models import get_llm_config

logger = logging.getLogger(__name__)

TaskType = Literal["generation", "word_translation", "sentence_translation"]


@dataclass
class ResolvedModel:
    """Resolved model configuration ready for use."""
    model_id: str
    provider: str
    base_url: str
    api_key: Optional[str]
    max_tokens: Optional[int]
    extra_params: dict
    source: str  # "user", "system", "subscription", "fallback"
    config_id: Optional[int] = None  # ID of UserModelConfig if from DB


class UserModelService:
    """Service for managing user's LLM model configurations."""
    
    def __init__(self):
        self.llm_config = get_llm_config()
    
    # ==================== CRUD Operations ====================
    
    def list_models(self, db: Session, account_id: int, include_inactive: bool = False) -> List[UserModelConfig]:
        """List all models for a user."""
        query = db.query(UserModelConfig).filter(UserModelConfig.account_id == account_id)
        if not include_inactive:
            query = query.filter(UserModelConfig.is_active == True)
        return query.order_by(UserModelConfig.priority.asc(), UserModelConfig.created_at.asc()).all()
    
    def get_model(self, db: Session, account_id: int, model_config_id: int) -> Optional[UserModelConfig]:
        """Get a specific model config by ID."""
        return db.query(UserModelConfig).filter(
            UserModelConfig.id == model_config_id,
            UserModelConfig.account_id == account_id
        ).first()
    
    def create_model(
        self,
        db: Session,
        account_id: int,
        display_name: str,
        provider: str,
        model_id: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        capabilities: Optional[List[str]] = None,
        extra_params: Optional[dict] = None,
        use_for_generation: bool = False,
        use_for_word_translation: bool = False,
        use_for_sentence_translation: bool = False,
        priority: int = 100,
    ) -> UserModelConfig:
        """Create a new user-defined model configuration."""
        config = UserModelConfig(
            account_id=account_id,
            display_name=display_name,
            provider=provider,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            source="user",
            is_editable=True,
            is_key_visible=True,
            max_tokens=max_tokens,
            capabilities=capabilities or ["text"],
            extra_params=extra_params or {},
            use_for_generation=use_for_generation,
            use_for_word_translation=use_for_word_translation,
            use_for_sentence_translation=use_for_sentence_translation,
            priority=priority,
            is_active=True,
        )
        db.add(config)
        db.commit()
        db.refresh(config)
        logger.info(f"Created user model config: {config.id} for account {account_id}")
        return config
    
    def update_model(
        self,
        db: Session,
        account_id: int,
        model_config_id: int,
        **updates
    ) -> Optional[UserModelConfig]:
        """Update a model configuration. Only user models are fully editable."""
        config = self.get_model(db, account_id, model_config_id)
        if not config:
            return None
        
        # Check editability
        if not config.is_editable:
            # For system models, only display_name can be changed
            allowed_fields = {"display_name"}
            updates = {k: v for k, v in updates.items() if k in allowed_fields}
        
        # Never allow changing source or is_editable
        updates.pop("source", None)
        updates.pop("is_editable", None)
        updates.pop("is_key_visible", None)
        
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        db.commit()
        db.refresh(config)
        logger.info(f"Updated model config: {config.id}")
        return config
    
    def delete_model(self, db: Session, account_id: int, model_config_id: int) -> bool:
        """Delete a model configuration. Only user models can be deleted."""
        config = self.get_model(db, account_id, model_config_id)
        if not config:
            return False
        
        if config.source != "user":
            logger.warning(f"Cannot delete non-user model config: {config.id}")
            return False
        
        db.delete(config)
        db.commit()
        logger.info(f"Deleted model config: {model_config_id}")
        return True
    
    def set_model_active(self, db: Session, account_id: int, model_config_id: int, is_active: bool) -> bool:
        """Enable or disable a model."""
        config = self.get_model(db, account_id, model_config_id)
        if not config:
            return False
        
        config.is_active = is_active
        db.commit()
        return True
    
    # ==================== Task Assignment ====================
    
    def assign_model_to_task(
        self,
        db: Session,
        account_id: int,
        model_config_id: int,
        task: TaskType,
        exclusive: bool = True
    ) -> bool:
        """
        Assign a model to a specific task.
        If exclusive=True, unassign all other models from this task first.
        """
        config = self.get_model(db, account_id, model_config_id)
        if not config:
            return False
        
        task_field = f"use_for_{task}"
        
        if exclusive:
            # Unassign all other models from this task
            db.query(UserModelConfig).filter(
                UserModelConfig.account_id == account_id,
                UserModelConfig.id != model_config_id
            ).update({task_field: False})
        
        setattr(config, task_field, True)
        db.commit()
        logger.info(f"Assigned model {model_config_id} to task {task}")
        return True
    
    def get_assigned_model(self, db: Session, account_id: int, task: TaskType) -> Optional[UserModelConfig]:
        """Get the model assigned to a specific task."""
        task_field = f"use_for_{task}"
        return db.query(UserModelConfig).filter(
            UserModelConfig.account_id == account_id,
            UserModelConfig.is_active == True,
            getattr(UserModelConfig, task_field) == True
        ).order_by(UserModelConfig.priority.asc()).first()
    
    # ==================== Model Resolution ====================
    
    def resolve_model_for_task(self, db: Session, account_id: int, task: TaskType) -> ResolvedModel:
        """
        Resolve which model to use for a given task.
        
        Resolution order:
        1. User's assigned model for this task (from DB)
        2. Any active model in user's config (by priority)
        3. System fallback from llm_models.json
        """
        # Try user's assigned model
        config = self.get_assigned_model(db, account_id, task)
        if config:
            return self._config_to_resolved(config)
        
        # Try any active model
        models = self.list_models(db, account_id, include_inactive=False)
        if models:
            return self._config_to_resolved(models[0])
        
        # Fall back to system default
        return self._get_system_fallback()
    
    def _config_to_resolved(self, config: UserModelConfig) -> ResolvedModel:
        """Convert a UserModelConfig to a ResolvedModel."""
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
        
        # Get API key
        api_key = config.api_key
        if not api_key and config.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
        
        return ResolvedModel(
            model_id=config.model_id,
            provider=config.provider,
            base_url=base_url,
            api_key=api_key,
            max_tokens=config.max_tokens,
            extra_params=config.extra_params or {},
            source=config.source,
            config_id=config.id
        )
    
    def _get_system_fallback(self) -> ResolvedModel:
        """Get the system default model as fallback."""
        # Try to get from llm_models.json config
        default_model = self.llm_config.get_model_by_id(self.llm_config.default_model)
        if default_model:
            return ResolvedModel(
                model_id=default_model.model,
                provider="openrouter" if "openrouter" in default_model.base_url else "local",
                base_url=default_model.base_url,
                api_key=default_model.get_api_key(),
                max_tokens=default_model.max_tokens,
                extra_params={},
                source="fallback"
            )
        
        # Ultimate fallback
        return ResolvedModel(
            model_id="x-ai/grok-4.1-fast:free",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            max_tokens=32768,
            extra_params={},
            source="fallback"
        )
    
    # ==================== System Model Injection ====================
    
    def inject_system_models(self, db: Session, account_id: int, tier: str = "Free") -> List[UserModelConfig]:
        """
        Inject system-provided models for a user based on their tier.
        Called when user signs up or subscription changes.
        """
        injected = []
        
        # Get models allowed for this tier from config
        tier_models = self.llm_config.get_models_for_tier(tier)
        
        for i, model_cfg in enumerate(tier_models):
            # Check if already exists
            existing = db.query(UserModelConfig).filter(
                UserModelConfig.account_id == account_id,
                UserModelConfig.model_id == model_cfg.model,
                UserModelConfig.source == "system"
            ).first()
            
            if existing:
                continue
            
            # Create system model entry
            config = UserModelConfig(
                account_id=account_id,
                display_name=model_cfg.display_name,
                provider="openrouter" if "openrouter" in model_cfg.base_url else "local",
                model_id=model_cfg.model,
                base_url=model_cfg.base_url,
                api_key=None,  # System models use env vars
                source="system",
                is_editable=False,
                is_key_visible=False,
                max_tokens=model_cfg.max_tokens,
                capabilities=model_cfg.capabilities,
                extra_params={},
                # First model is default for all tasks
                use_for_generation=(i == 0),
                use_for_word_translation=(i == 0),
                use_for_sentence_translation=(i == 0),
                priority=i * 10,  # 0, 10, 20, ...
                is_active=True,
            )
            db.add(config)
            injected.append(config)
        
        if injected:
            db.commit()
            logger.info(f"Injected {len(injected)} system models for account {account_id}")
        
        return injected
    
    def ensure_user_has_models(self, db: Session, account_id: int, tier: str = "Free") -> None:
        """Ensure user has at least the system models available."""
        existing = db.query(UserModelConfig).filter(
            UserModelConfig.account_id == account_id
        ).first()
        
        if not existing:
            self.inject_system_models(db, account_id, tier)
    
    def sync_system_models_for_tier(self, db: Session, account_id: int, new_tier: str) -> dict:
        """
        Sync system models when user's subscription tier changes.
        
        - Removes system models no longer allowed for the new tier
        - Adds system models newly available for the new tier
        - Preserves user-defined models
        
        Returns dict with counts: {"added": N, "removed": M}
        """
        # Get models allowed for new tier
        allowed_models = self.llm_config.get_models_for_tier(new_tier)
        allowed_model_ids = {m.model for m in allowed_models}
        
        # Get current system models for user
        current_system = db.query(UserModelConfig).filter(
            UserModelConfig.account_id == account_id,
            UserModelConfig.source == "system"
        ).all()
        current_model_ids = {m.model_id for m in current_system}
        
        removed = 0
        added = 0
        
        # Remove models no longer allowed
        for config in current_system:
            if config.model_id not in allowed_model_ids:
                # Check if this model was assigned to any task
                was_assigned = (
                    config.use_for_generation or 
                    config.use_for_word_translation or 
                    config.use_for_sentence_translation
                )
                
                db.delete(config)
                removed += 1
                logger.info(f"Removed system model {config.model_id} from account {account_id} (tier downgrade)")
                
                # If it was assigned, reassign to first available model
                if was_assigned:
                    self._reassign_tasks_after_removal(db, account_id)
        
        # Add newly allowed models
        for i, model_cfg in enumerate(allowed_models):
            if model_cfg.model not in current_model_ids:
                # Check if user has ANY models for tasks
                has_generation = db.query(UserModelConfig).filter(
                    UserModelConfig.account_id == account_id,
                    UserModelConfig.use_for_generation == True,
                    UserModelConfig.is_active == True
                ).first() is not None
                
                config = UserModelConfig(
                    account_id=account_id,
                    display_name=model_cfg.display_name,
                    provider="openrouter" if "openrouter" in model_cfg.base_url else "local",
                    model_id=model_cfg.model,
                    base_url=model_cfg.base_url,
                    api_key=None,
                    source="system",
                    is_editable=False,
                    is_key_visible=False,
                    max_tokens=model_cfg.max_tokens,
                    capabilities=model_cfg.capabilities,
                    extra_params={},
                    # Only assign to tasks if no other model is assigned
                    use_for_generation=(not has_generation and i == 0),
                    use_for_word_translation=(not has_generation and i == 0),
                    use_for_sentence_translation=(not has_generation and i == 0),
                    priority=(i + len(current_system)) * 10,
                    is_active=True,
                )
                db.add(config)
                added += 1
                logger.info(f"Added system model {model_cfg.model} for account {account_id} (tier upgrade)")
        
        if added or removed:
            db.commit()
        
        return {"added": added, "removed": removed}
    
    def _reassign_tasks_after_removal(self, db: Session, account_id: int) -> None:
        """Reassign tasks to first available model after a model is removed."""
        # Find first active model
        first_model = db.query(UserModelConfig).filter(
            UserModelConfig.account_id == account_id,
            UserModelConfig.is_active == True
        ).order_by(UserModelConfig.priority.asc()).first()
        
        if not first_model:
            return
        
        # Check each task and assign if unassigned
        for task_field in ["use_for_generation", "use_for_word_translation", "use_for_sentence_translation"]:
            has_assigned = db.query(UserModelConfig).filter(
                UserModelConfig.account_id == account_id,
                UserModelConfig.is_active == True,
                getattr(UserModelConfig, task_field) == True
            ).first()
            
            if not has_assigned:
                setattr(first_model, task_field, True)
                logger.info(f"Reassigned {task_field} to model {first_model.id} for account {account_id}")


# Global instance
_user_model_service: Optional[UserModelService] = None


def get_user_model_service() -> UserModelService:
    """Get the global user model service instance."""
    global _user_model_service
    if _user_model_service is None:
        _user_model_service = UserModelService()
    return _user_model_service
