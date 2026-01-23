"""
Request/Response schemas for API interactions.

Migrated from CLI-Tuner V1.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ToolType(str, Enum):
    """Supported tool types."""
    BASH = "bash"
    GIT = "git"
    DOCKER = "docker"


class GenerationRequest(BaseModel):
    """Request for command generation."""
    query: str = Field(min_length=3, max_length=500)
    tool: Optional[ToolType] = None
    context: Optional[dict] = None


class ValidationResult(str, Enum):
    """Result of command validation."""
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"


class GenerationResponse(BaseModel):
    """Response from command generation."""
    command: str
    tool: ToolType
    validation_result: ValidationResult
    allowed: bool
    risk_score: int = Field(ge=0, le=100)
    rationale: str
    asi_ids: List[str] = []
    mitre_ids: List[str] = []


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    code: str
    details: Optional[dict] = None
