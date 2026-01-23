"""
SigmaHQ Loader — YAML rule parser for CommandRisk

Pre-loads SigmaHQ detection rules at engine initialization
for fast pattern matching during validation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class SigmaRule:
    """Parsed SigmaHQ rule."""
    id: str
    title: str
    description: str
    status: str
    logsource: dict
    detection: dict
    level: str
    tags: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    
    @property
    def mitre_ids(self) -> List[str]:
        """Extract MITRE ATT&CK IDs from tags."""
        return [t.split(".")[-1] for t in self.tags if t.startswith("attack.")]


class SigmaLoader:
    """
    Loads and parses SigmaHQ YAML rules.
    
    Best Practice: Pre-load at engine initialization for speed + flexibility.
    """
    
    def __init__(self, rules_dir: Optional[Path] = None):
        self.rules_dir = rules_dir or Path("commandrisk/rules/sigma")
        self.rules: List[SigmaRule] = []
        
        if self.rules_dir.exists():
            self._load_rules()
    
    def _load_rules(self):
        """Parse all YAML files in rules directory."""
        for yaml_file in self.rules_dir.glob("**/*.yml"):
            try:
                rule = self._parse_rule(yaml_file)
                if rule:
                    self.rules.append(rule)
            except Exception as e:
                print(f"Warning: Failed to parse {yaml_file}: {e}")
    
    def _parse_rule(self, yaml_file: Path) -> Optional[SigmaRule]:
        """Parse a single Sigma YAML file."""
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not data or not isinstance(data, dict):
            return None
        
        return SigmaRule(
            id=data.get("id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            status=data.get("status", "experimental"),
            logsource=data.get("logsource", {}),
            detection=data.get("detection", {}),
            level=data.get("level", "medium"),
            tags=data.get("tags", []),
            references=data.get("references", [])
        )
    
    def get_rules_for_category(self, category: str) -> List[SigmaRule]:
        """Get rules matching a logsource category."""
        return [r for r in self.rules if r.logsource.get("category") == category]
    
    def get_rules_by_level(self, level: str) -> List[SigmaRule]:
        """Get rules of a specific severity level."""
        return [r for r in self.rules if r.level == level]
    
    def get_linux_rules(self) -> List[SigmaRule]:
        """Get rules applicable to Linux/shell commands."""
        return [
            r for r in self.rules 
            if r.logsource.get("product") in ["linux", "unix"]
            or r.logsource.get("category") in ["process_creation", "shell"]
        ]
