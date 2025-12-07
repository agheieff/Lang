"""
System services consolidating LLM configuration, usage tracking, and session management.
"""

from __future__ import annotations

import logging
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

from sqlalchemy.orm import Session
from fastapi import Request, HTTPException
from server.models import (
    Account,
    LLMModel,
    UserModelConfig,
    UsageTracking,
    GenerationLog,
    TranslationLog,
    LLMRequestLog,
    SubscriptionTier,
)

logger = logging.getLogger(__name__)


# LLM Configuration Service
class LLMConfigService:
    """Manages LLM model configuration and availability."""
    
    def __init__(self):
        self.models_config_path = Path(__file__).parent.parent / "llm" / "models.json"
        self._models_cache: Optional[Dict] = None
        self._cache_timestamp: Optional[datetime] = None
    
    def load_models_config(self) -> Dict:
        """Load LLM models configuration from JSON file."""
        try:
            with open(self.models_config_path, "r") as f:
                config = json.load(f)
            
            self._models_cache = config
            self._cache_timestamp = datetime.now(timezone.utc)
            return config
            
        except Exception as e:
            logger.error(f"Error loading models config: {e}")
            # Return default config
            return {
                "models": [],
                "default_model": "grok-fast-free",
                "fallback_chain": ["grok-fast-free"],
                "provider_configs": {}
            }
    
    def get_models_config(self, force_reload: bool = False) -> Dict:
        """Get models config, with caching."""
        if (force_reload or 
            self._models_cache is None or 
            self._cache_timestamp is None or
            (datetime.now(timezone.utc) - self._cache_timestamp) > timedelta(minutes=5)):
            return self.load_models_config()
        
        return self._models_cache
    
    def get_available_models_for_tier(
        self,
        tier: str,
    ) -> List[Dict]:
        """Get models available for a specific subscription tier."""
        config = self.get_models_config()
        
        available_models = []
        for model in config.get("models", []):
            allowed_tiers = model.get("allowed_tiers", [])
            if tier in allowed_tiers or "admin" in allowed_tiers:
                available_models.append(model)
        
        return available_models
    
    def get_default_model_for_tier(
        self,
        tier: str,
    ) -> Optional[Dict]:
        """Get default model for a tier."""
        available = self.get_available_models_for_tier(tier)
        config = self.get_models_config()
        default_id = config.get("default_model", "grok-fast-free")
        
        for model in available:
            if model.get("id") == default_id:
                return model
        
        # Return first available if default not found
        return available[0] if available else None
    
    def get_preferred_model_for_task(
        self,
        db: Session,
        account_id: int,
        task: str,  # generation, word_translation, sentence_translation
    ) -> Optional[UserModelConfig]:
        """Get user's preferred model for a specific task."""
        
        # Look for user assignment
        task_field_map = {
            "generation": "use_for_generation",
            "word_translation": "use_for_word_translation", 
            "sentence_translation": "use_for_sentence_translation",
        }
        
        field_name = task_field_map.get(task)
        if not field_name:
            return None
        
        # Get user's active models for this task
        assigned = db.query(UserModelConfig).filter(
            UserModelConfig.account_id == account_id,
            UserModelConfig.is_active == True,
            getattr(UserModelConfig, field_name) == True,
        ).order_by(UserModelConfig.priority.asc()).first()
        
        if assigned:
            return assigned
        
        # Fall back to system default
        account = db.query(Account).filter(Account.id == account_id).first()
        if account:
            tier = account.subscription_tier or "Free"
            default_model = self.get_default_model_for_tier(tier)
            if default_model:
                # Create temporary default config
                return UserModelConfig(
                    display_name=default_model.get("display_name"),
                    provider="openrouter",
                    model_id=default_model.get("model"),
                    base_url=default_model.get("base_url"),
                    source="system",
                    is_editable=False,
                    is_key_visible=False,
                    is_active=True,
                )
        
        return None
    
    def get_fallback_chain(self) -> List[str]:
        """Get model fallback chain for retry logic."""
        config = self.get_models_config()
        return config.get("fallback_chain", [])


# Usage Service
class UsageService:
    """Tracks API usage and enforces tier limits."""
    
    def __init__(self):
        self.tier_limits = {
            "Free": {"texts_per_month": 10, "chars_per_month": 5000},
            "Standard": {"texts_per_month": 100, "chars_per_month": 50000},
            "Pro": {"texts_per_month": 500, "chars_per_month": 250000},
            "Pro+": {"texts_per_month": 2000, "chars_per_month": 1000000},
            "BYOK": {"texts_per_month": 10000, "chars_per_month": 5000000},
        }
    
    def track_generation_usage(
        self,
        db: Session,
        account_id: int,
        text_id: int,
        chars_generated: int,
    ) -> bool:
        """Track usage and check limits."""
        try:
            # Get current month tracking
            now = datetime.now(timezone.utc)
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            usage = db.query(UsageTracking).filter(
                UsageTracking.account_id == account_id,
                UsageTracking.period_start == period_start,
            ).first()
            
            if not usage:
                usage = UsageTracking(
                    account_id=account_id,
                    period_start=period_start,
                    texts_generated=0,
                    chars_generated=0,
                )
                db.add(usage)
            
            # Check limits
            account = db.query(Account).filter(Account.id == account_id).first()
            if not account:
                return False
            
            tier = account.subscription_tier or "Free"
            limits = self.tier_limits.get(tier, self.tier_limits["Free"])
            
            # Check if we would exceed limits
            new_texts = usage.texts_generated + 1
            new_chars = usage.chars_generated + chars_generated
            
            if (new_texts > limits["texts_per_month"] or 
                new_chars > limits["chars_per_month"]):
                logger.warning(f"Account {account_id} exceeded tier limits")
                return False
            
            # Update usage
            usage.texts_generated = new_texts
            usage.chars_generated = new_chars
            usage.last_updated = now
            
            db.commit()
            
            logger.info(f"Tracked usage for account {account_id}: {new_texts} texts, {new_chars} chars")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking usage: {e}")
            db.rollback()
            return False
    
    def get_usage_stats(
        self,
        db: Session,
        account_id: int,
    ) -> Dict:
        """Get current usage statistics."""
        try:
            now = datetime.now(timezone.utc)
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            usage = db.query(UsageTracking).filter(
                UsageTracking.account_id == account_id,
                UsageTracking.period_start == period_start,
            ).first()
            
            if not usage:
                return {
                    "texts_generated": 0,
                    "chars_generated": 0,
                    "period_start": period_start,
                    "days_remaining": (period_start + timedelta(days=32) - now).days,
                }
            
            days_remaining = max(0, (period_start + timedelta(days=32) - now).days)
            
            return {
                "texts_generated": usage.texts_generated,
                "chars_generated": usage.chars_generated,
                "period_start": usage.period_start,
                "last_updated": usage.last_updated,
                "days_remaining": days_remaining,
            }
            
        except Exception as e:
            logger.error(f"Error getting usage stats: {e}")
            return {}
    
    def reset_monthly_usage(self) -> None:
        """Reset usage for new month (called by scheduler)."""
        # This would typically be called by a cron job
        pass


# Session Management Service
class SessionManagementService:
    """Manages user sessions and authentication state."""
    
    def __init__(self):
        self.session_timeout = timedelta(hours=24)
        self.active_sessions: Dict[str, Dict] = {}
    
    def create_session(
        self,
        request: Request,
        account_id: int,
        account_data: Dict,
    ) -> str:
        """Create a new session for user."""
        try:
            # Generate session token
            import secrets
            session_token = secrets.token_urlsafe(32)
            
            # Store session data
            session_data = {
                "account_id": account_id,
                "account_data": account_data,
                "created_at": datetime.now(timezone.utc),
                "last_accessed": datetime.now(timezone.utc),
                "ip_address": getattr(request, "client", {}).get("host"),
                "user_agent": request.headers.get("user-agent"),
            }
            
            self.active_sessions[session_token] = session_data
            
            # Set cookie
            response_data = {
                "session_token": session_token,
                "expires_at": session_data["created_at"] + self.session_timeout,
            }
            
            logger.info(f"Created session for account {account_id}")
            return session_token
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise
    
    def validate_session(
        self,
        session_token: str,
    ) -> Optional[Dict]:
        """Validate session token and return session data."""
        try:
            if session_token not in self.active_sessions:
                return None
            
            session_data = self.active_sessions[session_token]
            
            # Check timeout
            if datetime.now(timezone.utc) - session_data["last_accessed"] > self.session_timeout:
                self.invalidate_session(session_token)
                return None
            
            # Update last accessed
            session_data["last_accessed"] = datetime.now(timezone.utc)
            
            return session_data
            
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return None
    
    def invalidate_session(
        self,
        session_token: str,
    ) -> bool:
        """Invalidate a session."""
        try:
            if session_token in self.active_sessions:
                session_data = self.active_sessions[session_token]
                account_id = session_data.get("account_id")
                del self.active_sessions[session_token]
                
                logger.info(f"Invalidated session for account {account_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error invalidating session: {e}")
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        try:
            expired_sessions = []
            now = datetime.now(timezone.utc)
            
            for token, data in self.active_sessions.items():
                if now - data["last_accessed"] > self.session_timeout:
                    expired_sessions.append(token)
            
            for token in expired_sessions:
                self.invalidate_session(token)
            
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            return len(expired_sessions)
            
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
            return 0
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions."""
        return len(self.active_sessions)


# LLM Request Logging Service
class LLMRequestLogService:
    """Handles comprehensive logging of all LLM requests."""
    
    def log_request(
        self,
        db: Session,
        account_id: Optional[int],
        text_id: Optional[int],
        kind: str,
        provider: Optional[str],
        model: Optional[str],
        base_url: Optional[str],
        request_data: Dict,
        response_data: Optional[str],
        error_data: Optional[str],
        status: str = "ok",
    ) -> LLMRequestLog:
        """Log an LLM request."""
        try:
            log_entry = LLMRequestLog(
                account_id=account_id,
                text_id=text_id,
                kind=kind,
                provider=provider,
                model=model,
                base_url=base_url,
                status=status,
                request=request_data,
                response=response_data,
                error=error_data,
                created_at=datetime.now(timezone.utc),
            )
            
            db.add(log_entry)
            db.commit()
            db.refresh(log_entry)
            
            # Log to file for debugging
            if os.getenv("ARCADIA_DEBUG_LOGS"):
                self._log_to_file(log_entry)
            
            return log_entry
            
        except Exception as e:
            logger.error(f"Error logging LLM request: {e}")
            db.rollback()
            raise
    
    def get_request_history(
        self,
        db: Session,
        account_id: int,
        limit: int = 100,
        kind: Optional[str] = None,
    ) -> List[LLMRequestLog]:
        """Get LLM request history for account."""
        try:
            query = db.query(LLMRequestLog).filter(
                LLMRequestLog.account_id == account_id,
            )
            
            if kind:
                query = query.filter(LLMRequestLog.kind == kind)
            
            return query.order_by(
                LLMRequestLog.created_at.desc(),
            ).limit(limit).all()
            
        except Exception as e:
            logger.error(f"Error getting request history: {e}")
            return []
    
    def _log_to_file(self, log_entry: LLMRequestLog) -> None:
        """Log request to file for debugging."""
        try:
            log_dir = Path.cwd() / "data" / "llm_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"llm_requests_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
            
            log_data = {
                "timestamp": log_entry.created_at.isoformat(),
                "account_id": log_entry.account_id,
                "text_id": log_entry.text_id,
                "kind": log_entry.kind,
                "provider": log_entry.provider,
                "model": log_entry.model,
                "status": log_entry.status,
                "request_size": len(str(log_entry.request)),
                "response_size": len(str(log_entry.response) if log_entry.response else ""),
                "error": log_entry.error,
            }
            
            with open(log_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")
                
        except Exception as e:
            logger.error(f"Error logging to file: {e}")


# Notification Service
class NotificationService:
    """Sends real-time notifications to clients via SSE."""
    
    def __init__(self):
        self.active_connections: Dict[int, Set[str]] = {}  # account_id -> set of connection_ids
    
    def register_connection(
        self,
        account_id: int,
        connection_id: str,
    ) -> None:
        """Register a client connection."""
        if account_id not in self.active_connections:
            self.active_connections[account_id] = set()
        
        self.active_connections[account_id].add(connection_id)
        logger.info(f"Registered connection {connection_id} for account {account_id}")
    
    def unregister_connection(
        self,
        account_id: int,
        connection_id: str,
    ) -> None:
        """Unregister a client connection."""
        if account_id in self.active_connections:
            self.active_connections[account_id].discard(connection_id)
            
            if not self.active_connections[account_id]:
                del self.active_connections[account_id]
        
        logger.info(f"Unregistered connection {connection_id} for account {account_id}")
    
    async def notify_text_ready(
        self,
        account_id: int,
        text_id: int,
    ) -> None:
        """Notify client that a text is ready for reading."""
        try:
            notification = {
                "type": "text_ready",
                "text_id": text_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            await self._send_notification(account_id, notification)
            
        except Exception as e:
            logger.error(f"Error notifying text ready: {e}")
    
    async def notify_generation_progress(
        self,
        account_id: int,
        text_id: int,
        stage: str,
        progress: float,
    ) -> None:
        """Notify client about generation progress."""
        try:
            notification = {
                "type": "generation_progress",
                "text_id": text_id,
                "stage": stage,
                "progress": progress,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            await self._send_notification(account_id, notification)
            
        except Exception as e:
            logger.error(f"Error notifying generation progress: {e}")
    
    async def _send_notification(
        self,
        account_id: int,
        notification: Dict,
    ) -> None:
        """Send notification to all active connections for account."""
        # This would integrate with the SSE handler implementation
        # For now, we just log the notification
        logger.info(f"Sending notification to account {account_id}: {notification}")


# Service instances
_llm_config_service = None
_usage_service = None
_session_service = None
_llm_log_service = None
_notification_service = None


def get_llm_config_service() -> LLMConfigService:
    """Get the LLM config service instance."""
    global _llm_config_service
    if _llm_config_service is None:
        _llm_config_service = LLMConfigService()
    return _llm_config_service


def get_usage_service() -> UsageService:
    """Get the usage service instance."""
    global _usage_service
    if _usage_service is None:
        _usage_service = UsageService()
    return _usage_service


def get_session_service() -> SessionManagementService:
    """Get the session service instance."""
    global _session_service
    if _session_service is None:
        _session_service = SessionManagementService()
    return _session_service


def get_llm_log_service() -> LLMRequestLogService:
    """Get the LLM log service instance."""
    global _llm_log_service
    if _llm_log_service is None:
        _llm_log_service = LLMRequestLogService()
    return _llm_log_service


def get_notification_service() -> NotificationService:
    """Get the notification service instance."""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service
