"""
Secure Wrapper — Command interception and execution control

The final gate before any command is executed. Validates through
CommandRisk engine and logs all decisions for audit.
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from .engine import CommandRiskEngine, EngineResponse, ValidationResult


@dataclass
class ExecutionResult:
    """Result of a wrapped execution attempt."""
    allowed: bool
    command: str
    executed: bool
    output: Optional[str]
    error: Optional[str]
    validation: EngineResponse
    timestamp: datetime


class SecureWrapper:
    """
    Secure command execution wrapper.
    
    Intercepts every command, validates through CommandRisk,
    and only allows execution if all checks pass.
    """
    
    def __init__(self, engine: CommandRiskEngine = None, dry_run: bool = True):
        self.engine = engine or CommandRiskEngine()
        self.dry_run = dry_run  # Never execute by default
        self.execution_log: list[ExecutionResult] = []
    
    def wrap(self, command: str, context: dict = None) -> ExecutionResult:
        """
        Wrap a command for secure execution.
        
        Args:
            command: The command to potentially execute
            context: Execution context (user, environment, etc.)
        
        Returns:
            ExecutionResult with validation details
        """
        context = context or {}
        timestamp = datetime.now()
        
        # Validate through CommandRisk
        validation = self.engine.validate(command, context)
        
        # Log the attempt
        result = ExecutionResult(
            allowed=validation.allowed,
            command=command,
            executed=False,
            output=None,
            error=None,
            validation=validation,
            timestamp=timestamp
        )
        
        if not validation.allowed:
            result.error = f"Blocked: {validation.rationale}"
            self._log_execution(result)
            return result
        
        # Execute if allowed and not in dry_run mode
        if not self.dry_run:
            result = self._execute(command, result)
        else:
            result.output = "[DRY RUN] Command validated but not executed"
        
        self._log_execution(result)
        return result
    
    def _execute(self, command: str, result: ExecutionResult) -> ExecutionResult:
        """Actually execute the command (use with extreme caution)."""
        import subprocess
        
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            result.executed = True
            result.output = proc.stdout
            result.error = proc.stderr if proc.returncode != 0 else None
        except subprocess.TimeoutExpired:
            result.error = "Command timed out after 30 seconds"
        except Exception as e:
            result.error = f"Execution error: {str(e)}"
        
        return result
    
    def _log_execution(self, result: ExecutionResult):
        """Log execution attempt for audit."""
        self.execution_log.append(result)
        
        # TODO: Send to W&B SecureCLI-Monitoring
        # wandb.log({
        #     "command": result.command,
        #     "allowed": result.allowed,
        #     "executed": result.executed,
        #     "asi_id": result.validation.primary_asi_id,
        #     "mitre_id": result.validation.primary_mitre_id,
        #     "risk_score": result.validation.risk_score,
        # })
    
    def get_audit_log(self) -> list[dict]:
        """Get audit log in serializable format."""
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "command": r.command,
                "allowed": r.allowed,
                "executed": r.executed,
                "rationale": r.validation.rationale,
                "asi_id": r.validation.primary_asi_id,
                "mitre_id": r.validation.primary_mitre_id,
                "risk_score": r.validation.risk_score,
            }
            for r in self.execution_log
        ]
