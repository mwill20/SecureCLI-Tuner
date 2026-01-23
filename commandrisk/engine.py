"""
CommandRisk Engine — Main validation orchestrator

Runs commands through all three guardrails (Deterministic, Semantic, Policy)
and returns a unified validation result with OWASP ASI attribution.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from .guardrails.deterministic import DeterministicGuardrail
from .guardrails.semantic import SemanticGuardrail
from .guardrails.policy import PolicyGuardrail


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


class CommandRiskEngine:
    """
    Main CommandRisk validation engine.
    
    Orchestrates three guardrails:
    1. Deterministic — Regex patterns and blocklists
    2. Semantic — Intent-to-action alignment via ML
    3. Policy — OWASP ASI compliance enforcement
    """
    
    def __init__(self, sigma_rules_dir: str = None, mitre_mappings_dir: str = None):
        self.deterministic = DeterministicGuardrail()
        self.semantic = SemanticGuardrail()
        self.policy = PolicyGuardrail()
        
        # TODO: Load SigmaHQ rules
        # TODO: Load MITRE ATT&CK mappings
    
    def validate(self, command: str, context: dict = None) -> EngineResponse:
        """
        Validate a command through all guardrails.
        
        Args:
            command: The CLI command to validate
            context: Optional context (user, environment, etc.)
        
        Returns:
            EngineResponse with allow/block decision and attribution
        """
        context = context or {}
        responses: List[GuardrailResponse] = []
        
        # Layer 1: Deterministic
        det_response = self.deterministic.validate(command, context)
        responses.append(det_response)
        
        if det_response.result == ValidationResult.BLOCK:
            return self._build_response(command, responses, blocked_by="deterministic")
        
        # Layer 2: Semantic
        sem_response = self.semantic.validate(command, context)
        responses.append(sem_response)
        
        if sem_response.result == ValidationResult.BLOCK:
            return self._build_response(command, responses, blocked_by="semantic")
        
        # Layer 3: Policy
        pol_response = self.policy.validate(command, context)
        responses.append(pol_response)
        
        if pol_response.result == ValidationResult.BLOCK:
            return self._build_response(command, responses, blocked_by="policy")
        
        # All guardrails passed
        return self._build_response(command, responses, blocked_by=None)
    
    def _build_response(
        self, 
        command: str, 
        responses: List[GuardrailResponse], 
        blocked_by: Optional[str]
    ) -> EngineResponse:
        """Build unified engine response."""
        allowed = blocked_by is None
        
        # Collect all ASI and MITRE IDs
        all_asi = []
        all_mitre = []
        for r in responses:
            all_asi.extend(r.asi_ids)
            all_mitre.extend(r.mitre_ids)
        
        # Calculate risk score (0-100)
        risk_score = 0
        for r in responses:
            if r.result == ValidationResult.BLOCK:
                risk_score = 100
                break
            elif r.result == ValidationResult.WARN:
                risk_score = max(risk_score, 70)
        
        # Build rationale
        if blocked_by:
            blocking_response = next(r for r in responses if r.guardrail == blocked_by)
            rationale = f"Blocked by {blocked_by}: {blocking_response.rationale}"
        else:
            rationale = "Command passed all security checks"
        
        return EngineResponse(
            allowed=allowed,
            command=command,
            final_result=ValidationResult.BLOCK if not allowed else ValidationResult.ALLOW,
            guardrail_responses=responses,
            primary_asi_id=all_asi[0] if all_asi else None,
            primary_mitre_id=all_mitre[0] if all_mitre else None,
            rationale=rationale,
            risk_score=risk_score
        )
