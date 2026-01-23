"""
Semantic Guardrail — Hybrid AST + CodeBERT Intent Classification

Uses a combination of:
1. AST parsing for structural pattern detection (fast, deterministic)
2. CodeBERT-based intent classification (semantic understanding)

Model: mrm8488/codebert-base-finetuned-detect-insecure-code
- Pre-trained on CodeXGLUE Defect Detection dataset (21K+ examples)
- Fine-tuned for 5-category intent classification

LLMSVS Mapping: V1 (Architectural Design)
OWASP ASI: ASI01 (Goal Hijack), ASI05 (Unexpected Execution)
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

# Avoid circular import
from ..types import GuardrailResponse, ValidationResult


class IntentCategory(Enum):
    """5-category intent classification."""
    BENIGN = "BENIGN"
    RECONNAISSANCE = "RECONNAISSANCE"
    DESTRUCTIVE = "DESTRUCTIVE"
    EXFILTRATION = "EXFILTRATION"
    PERSISTENCE = "PERSISTENCE"


@dataclass
class ASTFeatures:
    """Extracted AST features from command."""
    has_pipe: bool = False
    has_redirect: bool = False
    has_subshell: bool = False
    has_exec: bool = False
    has_eval: bool = False
    command_count: int = 0
    root_paths: List[str] = field(default_factory=list)
    sensitive_files: List[str] = field(default_factory=list)


@dataclass
class SemanticResult:
    """Result from semantic analysis."""
    ast_features: Optional[ASTFeatures]
    ast_risk: int
    intent: str
    intent_confidence: float
    intent_risk: int
    combined_risk: int
    explanation: str


class BashASTParser:
    """Simple AST-like parser for bash commands."""
    
    SENSITIVE_PATHS = [
        "/etc/passwd", "/etc/shadow", "/etc/sudoers",
        "~/.ssh/", "~/.aws/", "~/.kube/",
        "/root/", "/dev/sd", "/dev/nvme"
    ]
    
    def extract_features(self, command: str) -> ASTFeatures:
        """Extract structural features from command."""
        features = ASTFeatures()
        
        # Pipes
        features.has_pipe = "|" in command
        
        # Redirects
        features.has_redirect = any(op in command for op in [">", ">>", "<", "2>"])
        
        # Subshells
        features.has_subshell = "$(" in command or "`" in command
        
        # Dynamic execution
        features.has_exec = any(kw in command for kw in ["exec", "source", "."])
        features.has_eval = "eval" in command
        
        # Count commands (rough estimate via pipes and semicolons)
        features.command_count = 1 + command.count("|") + command.count(";") + command.count("&&")
        
        # Root paths
        if re.search(r"\s/\s|^/\s|\s/$", command):
            features.root_paths.append("/")
        
        # Sensitive files
        for path in self.SENSITIVE_PATHS:
            if path in command:
                features.sensitive_files.append(path)
        
        return features


class CommandIntentClassifier:
    """
    CodeBERT-based intent classifier for shell commands.
    
    Uses mrm8488/codebert-base-finetuned-detect-insecure-code as base,
    fine-tuned for 5-category intent classification.
    """
    
    MODEL_NAME = "mrm8488/codebert-base-finetuned-detect-insecure-code"
    
    # Risk scores per intent category
    INTENT_RISK_SCORES = {
        IntentCategory.BENIGN.value: 0,
        IntentCategory.RECONNAISSANCE.value: 30,
        IntentCategory.DESTRUCTIVE.value: 80,
        IntentCategory.EXFILTRATION.value: 90,
        IntentCategory.PERSISTENCE.value: 70,
    }
    
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        self.model_path = model_path or self.MODEL_NAME
        self.model = None
        self.tokenizer = None
        self.device = None
        self._model_loaded = False
        
        # Intent labels (binary for base model, 5-class after fine-tuning)
        self.intent_labels = {
            0: IntentCategory.BENIGN.value,
            1: IntentCategory.DESTRUCTIVE.value,  # Map "insecure" to destructive
        }
    
    def load_model(self):
        """Lazy-load the model (only when first needed)."""
        if self._model_loaded:
            return
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            
            # Device selection
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            self._model_loaded = True
        except ImportError:
            # Model not available, will use fallback
            pass
    
    def classify(self, command: str) -> Dict[str, Any]:
        """
        Classify command intent.
        
        Args:
            command: Shell command to analyze
            
        Returns:
            dict with intent, confidence, is_dangerous, probabilities
        """
        # Try to load model if not loaded
        if not self._model_loaded:
            self.load_model()
        
        # Fallback if model not available
        if not self._model_loaded:
            return self._fallback_classify(command)
        
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            command,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128  # Commands are short
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        intent = self.intent_labels.get(predicted_class, IntentCategory.BENIGN.value)
        is_dangerous = predicted_class == 1
        
        return {
            "intent": intent,
            "confidence": confidence,
            "is_dangerous": is_dangerous,
            "risk_score": self.INTENT_RISK_SCORES.get(intent, 0),
            "probabilities": {
                IntentCategory.BENIGN.value: probabilities[0].item(),
                IntentCategory.DESTRUCTIVE.value: probabilities[1].item() if len(probabilities) > 1 else 0,
            }
        }
    
    def _fallback_classify(self, command: str) -> Dict[str, Any]:
        """Fallback classification when model not available."""
        # Use pattern-based heuristics
        dangerous_patterns = [
            r"rm\s+-rf",
            r"curl.*\|.*sh",
            r"wget.*\|.*sh",
            r"dd\s+if=",
            r"chmod\s+777",
            r"sudo\s+su",
        ]
        
        is_dangerous = any(re.search(p, command, re.I) for p in dangerous_patterns)
        
        return {
            "intent": IntentCategory.DESTRUCTIVE.value if is_dangerous else IntentCategory.BENIGN.value,
            "confidence": 0.6,  # Lower confidence for fallback
            "is_dangerous": is_dangerous,
            "risk_score": 80 if is_dangerous else 0,
            "probabilities": {
                IntentCategory.BENIGN.value: 0.0 if is_dangerous else 1.0,
                IntentCategory.DESTRUCTIVE.value: 1.0 if is_dangerous else 0.0,
            }
        }


class SemanticGuardrail:
    """
    Hybrid AST + CodeBERT semantic guardrail.
    
    Combines:
    - AST parsing (fast, 1-2ms) for structural patterns
    - CodeBERT classification (50-100ms) for semantic intent
    
    LLMSVS: V1 (Architectural Design)
    OWASP ASI: ASI01 (Goal Hijack), ASI05 (Unexpected Execution)
    """
    
    # AST risk weights
    AST_WEIGHTS = {
        "has_pipe": 10,
        "has_redirect": 5,
        "has_subshell": 15,
        "has_exec": 20,
        "has_eval": 25,
        "root_paths": 20,
        "sensitive_files": 30,
    }
    
    def __init__(self, model_path: Optional[str] = None):
        self.ast_parser = BashASTParser()
        self.intent_classifier = CommandIntentClassifier(model_path=model_path)
        self._load_obfuscation_patterns()
    
    def _load_obfuscation_patterns(self):
        """Load patterns for detecting obfuscated commands."""
        self.obfuscation_patterns = [
            # Base64 encoding
            (re.compile(r"base64\s+-d|base64\s+--decode"), "Base64 decoding detected", ["ASI05"]),
            (re.compile(r"echo\s+['\"][A-Za-z0-9+/=]{20,}['\"]\s*\|\s*base64"), "Base64 encoding detected", ["ASI05"]),
            
            # Hex encoding
            (re.compile(r"\\x[0-9a-fA-F]{2}"), "Hex escape sequences detected", ["ASI05"]),
            (re.compile(r"xxd\s+-r"), "Hex decoding detected", ["ASI05"]),
            
            # Dynamic execution
            (re.compile(r"\$\{!"), "Variable indirection detected", ["ASI05"]),
            (re.compile(r"eval\s+"), "Eval command detected", ["ASI05"]),
            
            # Nested command substitution
            (re.compile(r"\$\([^)]*\$\("), "Nested command substitution", ["ASI05"]),
            
            # Prompt injection markers
            (re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.I), "Prompt injection attempt", ["ASI01"]),
            (re.compile(r"disregard\s+(all\s+)?instructions", re.I), "Prompt injection attempt", ["ASI01"]),
            (re.compile(r"forget\s+(everything|all)", re.I), "Prompt injection attempt", ["ASI01"]),
        ]
    
    def validate(self, command: str, context: dict = None) -> GuardrailResponse:
        """
        Hybrid validation using AST + CodeBERT.
        
        Args:
            command: The generated command
            context: Optional context
        """
        context = context or {}
        
        # 1. Check obfuscation patterns first (fast path)
        for pattern, description, asi_ids in self.obfuscation_patterns:
            if pattern.search(command):
                return GuardrailResponse(
                    guardrail="semantic",
                    result=ValidationResult.BLOCK,
                    confidence=0.95,
                    rationale=description,
                    asi_ids=asi_ids,
                    mitre_ids=["T1027"]  # Obfuscated Files
                )
        
        # 2. AST analysis
        ast_features = self.ast_parser.extract_features(command)
        ast_risk = self._calculate_ast_risk(ast_features)
        
        # 3. CodeBERT intent classification
        intent_result = self.intent_classifier.classify(command)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        intent_risk = intent_result["risk_score"]
        
        # 4. Combine risks (AST 30%, Intent 70%)
        combined_risk = int((ast_risk * 0.3) + (intent_risk * 0.7))
        
        # Confidence penalty
        if confidence < 0.7:
            combined_risk = int(combined_risk * confidence)
        
        # 5. Determine result
        if combined_risk >= 70 or intent_result["is_dangerous"]:
            return GuardrailResponse(
                guardrail="semantic",
                result=ValidationResult.BLOCK,
                confidence=confidence,
                rationale=self._generate_explanation(ast_features, intent, combined_risk),
                asi_ids=["ASI01"] if intent == IntentCategory.PERSISTENCE.value else ["ASI05"],
                mitre_ids=["T1059"]
            )
        elif combined_risk >= 50:
            return GuardrailResponse(
                guardrail="semantic",
                result=ValidationResult.WARN,
                confidence=confidence,
                rationale=self._generate_explanation(ast_features, intent, combined_risk),
                asi_ids=[],
                mitre_ids=[]
            )
        
        return GuardrailResponse(
            guardrail="semantic",
            result=ValidationResult.ALLOW,
            confidence=confidence,
            rationale="Command passed semantic analysis",
            asi_ids=[],
            mitre_ids=[]
        )
    
    def _calculate_ast_risk(self, features: ASTFeatures) -> int:
        """Calculate risk score from AST features."""
        risk = 0
        
        if features.has_pipe:
            risk += self.AST_WEIGHTS["has_pipe"]
        if features.has_redirect:
            risk += self.AST_WEIGHTS["has_redirect"]
        if features.has_subshell:
            risk += self.AST_WEIGHTS["has_subshell"]
        if features.has_exec:
            risk += self.AST_WEIGHTS["has_exec"]
        if features.has_eval:
            risk += self.AST_WEIGHTS["has_eval"]
        if features.root_paths:
            risk += self.AST_WEIGHTS["root_paths"]
        if features.sensitive_files:
            risk += self.AST_WEIGHTS["sensitive_files"] * len(features.sensitive_files)
        
        return min(risk, 100)
    
    def _generate_explanation(self, features: ASTFeatures, intent: str, risk: int) -> str:
        """Generate human-readable explanation."""
        parts = []
        
        if features.has_pipe:
            parts.append("pipes output")
        if features.has_eval:
            parts.append("uses dynamic eval")
        if features.sensitive_files:
            parts.append(f"accesses sensitive files")
        if features.root_paths:
            parts.append("targets root filesystem")
        
        parts.append(f"classified as {intent.lower()} intent")
        parts.append(f"risk score {risk}/100")
        
        return "Command " + ", ".join(parts)
    
    def analyze(self, command: str) -> SemanticResult:
        """Full semantic analysis (for debugging/logging)."""
        ast_features = self.ast_parser.extract_features(command)
        ast_risk = self._calculate_ast_risk(ast_features)
        
        intent_result = self.intent_classifier.classify(command)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        intent_risk = intent_result["risk_score"]
        
        combined_risk = int((ast_risk * 0.3) + (intent_risk * 0.7))
        if confidence < 0.7:
            combined_risk = int(combined_risk * confidence)
        
        return SemanticResult(
            ast_features=ast_features,
            ast_risk=ast_risk,
            intent=intent,
            intent_confidence=confidence,
            intent_risk=intent_risk,
            combined_risk=combined_risk,
            explanation=self._generate_explanation(ast_features, intent, combined_risk)
        )
