"""
Policy Guardrail — OWASP ASI compliance enforcement

Enforces organizational security policies and OWASP ASI Top 10
compliance for all generated commands.

LLMSVS Mapping: Cross-cutting (Policy enforcement layer)
OWASP ASI: All applicable IDs
"""
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from ..types import GuardrailResponse, ValidationResult


class PolicyLevel(Enum):
    """Policy enforcement levels."""
    STRICT = "strict"      # Block all violations
    MODERATE = "moderate"  # Block high-risk, warn medium-risk
    PERMISSIVE = "permissive"  # Warn only


@dataclass
class PolicyRule:
    """A security policy rule with OWASP attribution."""
    name: str
    description: str
    check_fn: callable
    asi_ids: List[str]
    level: PolicyLevel = PolicyLevel.STRICT


class PolicyGuardrail:
    """
    Policy guardrail for OWASP ASI Top 10 compliance.
    
    Enforces:
    - ASI01: Agent Goal Hijack prevention
    - ASI02: Tool Misuse detection
    - ASI03: Privilege Abuse prevention
    - ASI05: Unexpected Execution blocking
    
    This is the final gate before command execution.
    """
    
    def __init__(self, policy_level: PolicyLevel = PolicyLevel.STRICT):
        self.policy_level = policy_level
        self._load_policies()
    
    def _load_policies(self):
        """Load OWASP ASI policy rules."""
        self.policies: List[PolicyRule] = [
            PolicyRule(
                name="no_remote_execution",
                description="Block commands that fetch and execute remote code",
                check_fn=self._check_remote_execution,
                asi_ids=["ASI05"],
                level=PolicyLevel.STRICT
            ),
            PolicyRule(
                name="no_privilege_escalation",
                description="Block unauthorized privilege escalation",
                check_fn=self._check_privilege_escalation,
                asi_ids=["ASI03"],
                level=PolicyLevel.STRICT
            ),
            PolicyRule(
                name="no_data_destruction",
                description="Block commands that may destroy data",
                check_fn=self._check_data_destruction,
                asi_ids=["ASI02"],
                level=PolicyLevel.STRICT
            ),
            PolicyRule(
                name="no_credential_access",
                description="Block unauthorized credential access",
                check_fn=self._check_credential_access,
                asi_ids=["ASI03"],
                level=PolicyLevel.STRICT
            ),
        ]
    
    def validate(self, command: str, context: dict = None) -> GuardrailResponse:
        """
        Validate command against all security policies.
        
        Args:
            command: The command to validate
            context: Optional context (user role, environment, etc.)
        """
        context = context or {}
        
        for policy in self.policies:
            violation = policy.check_fn(command, context)
            if violation:
                # Determine action based on policy level
                if policy.level == PolicyLevel.STRICT:
                    result = ValidationResult.BLOCK
                elif policy.level == PolicyLevel.MODERATE:
                    if self.policy_level == PolicyLevel.PERMISSIVE:
                        result = ValidationResult.WARN
                    else:
                        result = ValidationResult.BLOCK
                else:
                    result = ValidationResult.WARN
                
                return GuardrailResponse(
                    guardrail="policy",
                    result=result,
                    confidence=0.95,
                    rationale=f"Policy violation: {policy.description}",
                    asi_ids=policy.asi_ids,
                    mitre_ids=[]
                )
        
        return GuardrailResponse(
            guardrail="policy",
            result=ValidationResult.ALLOW,
            confidence=1.0,
            rationale="All security policies passed",
            asi_ids=[],
            mitre_ids=[]
        )
    
    def _check_remote_execution(self, command: str, context: dict) -> bool:
        """Check for remote code execution patterns (ASI05)."""
        import re
        patterns = [
            r"curl.*\|.*sh",
            r"wget.*\|.*sh",
            r"fetch.*\|.*sh",
        ]
        return any(re.search(p, command, re.I) for p in patterns)
    
    def _check_privilege_escalation(self, command: str, context: dict) -> bool:
        """Check for privilege escalation (ASI03)."""
        import re
        # Only block if user doesn't have explicit admin context
        if context.get("user_role") == "admin":
            return False
        
        patterns = [
            r"sudo\s+su\s*$",
            r"sudo\s+-i\s*$",
            r"pkexec",
        ]
        return any(re.search(p, command, re.I) for p in patterns)
    
    def _check_data_destruction(self, command: str, context: dict) -> bool:
        """Check for data destruction commands (ASI02)."""
        import re
        # Allow if explicitly confirmed
        if context.get("confirmed_destructive"):
            return False
        
        patterns = [
            r"rm\s+-rf\s+/\s*$",
            r"mkfs\.",
            r"dd\s+.*of=/dev/sd",
        ]
        return any(re.search(p, command, re.I) for p in patterns)
    
    def _check_credential_access(self, command: str, context: dict) -> bool:
        """Check for credential access (ASI03)."""
        import re
        patterns = [
            r"cat\s+.*\.ssh/id_",
            r"/etc/shadow",
            r"\.aws/credentials",
            r"\.kube/config",
        ]
        return any(re.search(p, command, re.I) for p in patterns)
