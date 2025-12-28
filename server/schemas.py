"""Standard error response schemas for API endpoints."""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response format."""

    status: str = Field("error", description="Always 'error' for errors")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details (optional)"
    )


class SuccessResponse(BaseModel):
    """Standard success response format."""

    status: str = Field("ok", description="Always 'ok' for success")
    message: Optional[str] = Field(None, description="Optional success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")


class ValidationError(BaseModel):
    """Validation error detail."""

    field: str
    message: str


class ValidationErrorResponse(BaseModel):
    """Response for validation errors."""

    status: str = "error"
    message: str = "Validation failed"
    errors: list[ValidationError] = []
