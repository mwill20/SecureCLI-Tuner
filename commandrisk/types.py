"""
CommandRisk Types — Shared dataclasses to avoid circular imports
"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ValidationResult(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"


@dataclass
class GuardrailResponse:
    """Response from a single guardrail."""
    guardrail: str
    result: ValidationResult
    confidence: float
    rationale: str
    asi_ids: List[str] = field(default_factory=list)
    mitre_ids: List[str] = field(default_factory=list)


@dataclass
class EngineResponse:
    """Unified response from CommandRisk engine."""
    allowed: bool
    command: str
    final_result: ValidationResult
    guardrail_responses: List[GuardrailResponse]
    primary_asi_id: Optional[str] = None
    primary_mitre_id: Optional[str] = None
    rationale: str = ""
    risk_score: int = 0
    blocked_by_layer: Optional[str] = None
