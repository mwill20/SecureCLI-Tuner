"""
Router — Tool selection for Bash, Git, or Docker

Uses router_system.txt prompt to classify natural language queries
and route to the appropriate domain-specific generator.
"""
from enum import Enum
from dataclasses import dataclass


class ToolType(Enum):
    BASH = "bash"
    GIT = "git"
    DOCKER = "docker"
    UNKNOWN = "unknown"


@dataclass
class RouterResult:
    tool: ToolType
    confidence: float
    reasoning: str


class ToolRouter:
    """Routes natural language queries to appropriate tool generators."""
    
    def __init__(self, model=None):
        self.model = model
        self._load_prompts()
    
    def _load_prompts(self):
        """Load router system prompt."""
        # TODO: Load from prompts/router_system.txt
        self.system_prompt = """You are a tool router. Classify the user query as:
- BASH: Shell commands, file operations, system administration
- GIT: Version control, commits, branches, repositories
- DOCKER: Container operations, images, volumes, networks

Respond with only the tool name."""
    
    def route(self, query: str) -> RouterResult:
        """Classify query and return routing decision."""
        query_lower = query.lower()
        
        # Simple keyword-based routing (placeholder for LLM routing)
        if any(kw in query_lower for kw in ["docker", "container", "image", "volume"]):
            return RouterResult(ToolType.DOCKER, 0.9, "Docker keywords detected")
        elif any(kw in query_lower for kw in ["git", "commit", "branch", "push", "pull", "merge"]):
            return RouterResult(ToolType.GIT, 0.9, "Git keywords detected")
        else:
            return RouterResult(ToolType.BASH, 0.7, "Default to Bash")
