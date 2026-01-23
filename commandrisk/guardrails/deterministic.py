"""
Deterministic Guardrail — Pattern-based blocking

Migrated from V1 patterns.py with 17 zero-tolerance regex patterns.
Each pattern is mapped to OWASP ASI and MITRE ATT&CK IDs.
"""
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Pattern

# Import from types to avoid circular imports
from ..types import GuardrailResponse, ValidationResult


@dataclass
class DangerousPattern:
    """A dangerous command pattern with security attribution."""
    pattern: str
    description: str
    asi_ids: List[str]
    mitre_ids: List[str]
    compiled: Pattern = field(init=False)
    
    def __post_init__(self):
        self.compiled = re.compile(self.pattern, re.IGNORECASE)


# 17 Zero-Tolerance Patterns (migrated from V1)
DANGEROUS_PATTERNS: List[DangerousPattern] = [
    # Destructive file operations
    DangerousPattern(
        pattern=r"rm\s+(-[rf]+\s+)*(/|/\*|\.\.|~)",
        description="Recursive deletion of root or home directory",
        asi_ids=["ASI02"],
        mitre_ids=["T1485"]  # Data Destruction
    ),
    DangerousPattern(
        pattern=r"rm\s+-rf\s+/",
        description="Force recursive deletion of root filesystem",
        asi_ids=["ASI02"],
        mitre_ids=["T1485"]
    ),
    
    # Fork bomb
    DangerousPattern(
        pattern=r":\(\)\{\s*:\|:\s*&\s*\};:",
        description="Fork bomb (denial of service)",
        asi_ids=["ASI02"],
        mitre_ids=["T1499"]  # Endpoint DoS
    ),
    
    # Disk destruction
    DangerousPattern(
        pattern=r"dd\s+if=/dev/(zero|random|urandom)\s+of=/dev/(sd|hd|nvme)",
        description="Overwrite disk with zeros/random data",
        asi_ids=["ASI02"],
        mitre_ids=["T1561"]  # Disk Wipe
    ),
    DangerousPattern(
        pattern=r"mkfs\s+",
        description="Format filesystem",
        asi_ids=["ASI02"],
        mitre_ids=["T1561"]
    ),
    
    # Privilege escalation
    DangerousPattern(
        pattern=r"chmod\s+777\s+/",
        description="Set world-writable permissions on root",
        asi_ids=["ASI03"],
        mitre_ids=["T1222"]  # File Permissions Modification
    ),
    DangerousPattern(
        pattern=r"chown\s+root\s+",
        description="Change ownership to root",
        asi_ids=["ASI03"],
        mitre_ids=["T1222"]
    ),
    DangerousPattern(
        pattern=r"sudo\s+su\s*$",
        description="Escalate to root shell",
        asi_ids=["ASI03"],
        mitre_ids=["T1548"]  # Abuse Elevation Control
    ),
    
    # Network exfiltration
    DangerousPattern(
        pattern=r"curl\s+.*\|\s*(ba)?sh",
        description="Pipe remote script to shell",
        asi_ids=["ASI05"],
        mitre_ids=["T1059.004"]  # Unix Shell
    ),
    DangerousPattern(
        pattern=r"wget\s+-O-?\s+.*\|\s*(ba)?sh",
        description="Download and execute remote script",
        asi_ids=["ASI05"],
        mitre_ids=["T1059.004"]
    ),
    
    # Credential exposure
    DangerousPattern(
        pattern=r"cat\s+.*\.ssh/(id_rsa|id_ed25519|authorized_keys)",
        description="Read SSH private keys",
        asi_ids=["ASI03"],
        mitre_ids=["T1552.004"]  # Private Keys
    ),
    DangerousPattern(
        pattern=r"printenv\s*\|\s*grep\s+(KEY|SECRET|TOKEN|PASSWORD)",
        description="Extract secrets from environment",
        asi_ids=["ASI03"],
        mitre_ids=["T1552.001"]  # Credentials in Files
    ),
    
    # History manipulation
    DangerousPattern(
        pattern=r"history\s+-c",
        description="Clear command history (anti-forensics)",
        asi_ids=["ASI02"],
        mitre_ids=["T1070.003"]  # Clear Command History
    ),
    
    # Cron persistence
    DangerousPattern(
        pattern=r"crontab\s+-r",
        description="Remove all cron jobs",
        asi_ids=["ASI02"],
        mitre_ids=["T1053.003"]  # Cron
    ),
    
    # Network manipulation
    DangerousPattern(
        pattern=r"iptables\s+-F",
        description="Flush all firewall rules",
        asi_ids=["ASI02"],
        mitre_ids=["T1562.004"]  # Disable Firewall
    ),
    
    # Docker escape
    DangerousPattern(
        pattern=r"docker\s+run\s+.*--privileged",
        description="Run privileged container (potential escape)",
        asi_ids=["ASI03"],
        mitre_ids=["T1611"]  # Escape to Host
    ),
    
    # Git credential theft
    DangerousPattern(
        pattern=r"git\s+config\s+.*credential",
        description="Access/modify git credentials",
        asi_ids=["ASI03"],
        mitre_ids=["T1552.001"]
    ),
]


class DeterministicGuardrail:
    """
    Deterministic guardrail using regex pattern matching.
    
    LLMSVS Mapping: V2 (Model Operating Environment)
    OWASP ASI: ASI02, ASI03
    """
    
    def __init__(self, custom_patterns: List[DangerousPattern] = None):
        self.patterns = DANGEROUS_PATTERNS.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)
    
    def validate(self, command: str, context: dict = None) -> GuardrailResponse:
        """Check command against all dangerous patterns."""
        for pattern in self.patterns:
            if pattern.compiled.search(command):
                return GuardrailResponse(
                    guardrail="deterministic",
                    result=ValidationResult.BLOCK,
                    confidence=1.0,
                    rationale=pattern.description,
                    asi_ids=pattern.asi_ids,
                    mitre_ids=pattern.mitre_ids
                )
        
        return GuardrailResponse(
            guardrail="deterministic",
            result=ValidationResult.ALLOW,
            confidence=1.0,
            rationale="No dangerous patterns detected",
            asi_ids=[],
            mitre_ids=[]
        )
    
    def is_dangerous(self, command: str) -> bool:
        """Quick check if command matches any dangerous pattern."""
        return any(p.compiled.search(command) for p in self.patterns)
