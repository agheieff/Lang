"""
Application-specific exception classes.

Provides structured exception handling for different types of errors.
"""

from typing import Optional


class ArcadiaError(Exception):
    """Base exception class for Arcadia application errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class GenerationError(ArcadiaError):
    """Raised when text generation fails."""
    
    def __init__(self, message: str, text_id: Optional[int] = None, account_id: Optional[int] = None, **kwargs):
        super().__init__(message, error_code="GENERATION_ERROR", **kwargs)
        self.text_id = text_id
        self.account_id = account_id


class TranslationError(ArcadiaError):
    """Raised when translation generation fails."""
    
    def __init__(self, message: str, text_id: Optional[int] = None, unit: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="TRANSLATION_ERROR", **kwargs)
        self.text_id = text_id
        self.unit = unit  # "word", "sentence", etc.


class QuotaExceededError(ArcadiaError):
    """Raised when user exceeds their usage quota."""
    
    def __init__(self, message: str, account_id: Optional[int] = None, quota_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="QUOTA_EXCEEDED", **kwargs)
        self.account_id = account_id
        self.quota_type = quota_type


class TextNotFoundError(ArcadiaError):
    """Raised when a requested text cannot be found."""
    
    def __init__(self, text_id: int, **kwargs):
        message = f"Text with ID {text_id} not found"
        super().__init__(message, error_code="TEXT_NOT_FOUND", **kwargs)
        self.text_id = text_id


class ProfileNotFoundError(ArcadiaError):
    """Raised when a user profile cannot be found."""
    
    def __init__(self, account_id: int, **kwargs):
        message = f"Profile for account {account_id} not found"
        super().__init__(message, error_code="PROFILE_NOT_FOUND", **kwargs)
        self.account_id = account_id


class TextNotReadyError(ArcadiaError):
    """Raised when a text is not ready for reading."""
    
    def __init__(self, text_id: int, reason: str = "Text not ready", **kwargs):
        message = f"Text {text_id} not ready: {reason}"
        super().__init__(message, error_code="TEXT_NOT_READY", **kwargs)
        self.text_id = text_id
        self.reason = reason


class DatabaseError(ArcadiaError):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DATABASE_ERROR", **kwargs)
        self.operation = operation


class ValidationError(ArcadiaError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[object] = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field = field
        self.value = value


class ConfigurationError(ArcadiaError):
    """Raised when application configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)
        self.config_key = config_key


class ExternalServiceError(ArcadiaError):
    """Raised when external service calls fail."""
    
    def __init__(
        self, 
        message: str, 
        service_name: Optional[str] = None, 
        status_code: Optional[int] = None, 
        **kwargs
    ):
        super().__init__(message, error_code="EXTERNAL_SERVICE_ERROR", **kwargs)
        self.service_name = service_name
        self.status_code = status_code


class SessionError(ArcadiaError):
    """Raised when session management operations fail."""
    
    def __init__(self, message: str, account_id: Optional[int] = None, session_id: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="SESSION_ERROR", **kwargs)
        self.account_id = account_id
        self.session_id = session_id


class CacheError(ArcadiaError):
    """Raised when cache operations fail."""
    
    def __init__(self, message: str, cache_key: Optional[str] = None, operation: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CACHE_ERROR", **kwargs)
        self.cache_key = cache_key
        self.operation = operation


# Utility functions for error handling
def handle_error(error: Exception, context: str = "", logger=None) -> ArcadiaError:
    """
    Convert a generic exception to an ArcadiaError with context.
    
    Args:
        error: The original exception
        context: Additional context about where the error occurred
        logger: Optional logger to log the error
    
    Returns:
        ArcadiaError instance
    """
    if logger:
        logger.error(f"{context}: {error}", exc_info=True)
    
    if isinstance(error, ArcadiaError):
        return error
    
    # Convert common errors to specific types
    error_message = f"{context}: {error}" if context else str(error)
    
    if "quota" in str(error).lower() or "limit" in str(error).lower():
        return QuotaExceededError(error_message)
    elif "database" in str(error).lower() or "connection" in str(error).lower():
        return DatabaseError(error_message)
    elif "validation" in str(error).lower() or "invalid" in str(error).lower():
        return ValidationError(error_message)
    else:
        return ArcadiaError(error_message, details={"original_error": str(error)})


def log_error(error: ArcadiaError, logger=None, level: str = "error"):
    """
    Log an ArcadiaError with structured information.
    
    Args:
        error: The error to log
        logger: Logger instance (uses default logger if None)
        level: Log level ("error", "warning", "info", "debug")
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    log_func = getattr(logger, level)
    log_func(
        f"{error.error_code}: {error.message}",
        extra={
            "error_code": error.error_code,
            "details": error.details,
            "context": {
                attr: getattr(error, attr, None)
                for attr in ['text_id', 'account_id', 'unit', 'service_name']
                if hasattr(error, attr)
            }
        }
    )
