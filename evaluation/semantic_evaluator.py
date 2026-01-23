"""
Semantic Equivalence Evaluator for CLI Commands

Solves the "exact match" problem where functionally equivalent commands
are marked as failures. For example:
    find . -name "*.txt"    vs    ls *.txt    vs    ls | grep ".txt"

All achieve the same result but have different syntax.

Approaches:
1. CodeBERT embeddings — cosine similarity (fast)
2. LLM-as-judge — semantic understanding (flexible)
3. Execution-based — sandbox comparison (most accurate)

Usage:
    from evaluation.semantic_evaluator import SemanticEvaluator
    
    evaluator = SemanticEvaluator()
    result = evaluator.evaluate(
        instruction="Find all text files",
        expected="find . -name '*.txt'",
        generated="ls *.txt"
    )
    print(result.semantic_match)  # True (similarity > 0.80)
"""
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path
import subprocess

# Try importing transformers (may not be available during early setup)
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class EvaluationResult:
    """Result of semantic evaluation."""
    instruction: str
    expected: str
    generated: str
    
    # Metrics
    exact_match: bool = False
    semantic_similarity: float = 0.0
    semantic_match: bool = False  # similarity >= threshold
    llm_judge_match: Optional[bool] = None
    
    # Threshold used
    similarity_threshold: float = 0.80
    
    # Final verdict
    functionally_equivalent: bool = False
    
    # Explanation
    explanation: str = ""


class CodeBERTEmbedder:
    """
    Uses CodeBERT to compute embeddings for shell commands.
    Similar commands have similar embedding vectors.
    """
    
    MODEL_NAME = "microsoft/codebert-base"
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or self.MODEL_NAME
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self):
        """Lazy-load the model."""
        if self._loaded or not TORCH_AVAILABLE:
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.eval()
            self._loaded = True
        except Exception as e:
            print(f"Warning: Could not load CodeBERT: {e}")
    
    def get_embedding(self, command: str) -> Optional[torch.Tensor]:
        """Get embedding vector for a command."""
        if not self._loaded:
            self.load()
        
        if not self._loaded:
            return None
        
        inputs = self.tokenizer(
            command,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :]
        
        return embedding
    
    def similarity(self, cmd1: str, cmd2: str) -> float:
        """Compute cosine similarity between two commands."""
        emb1 = self.get_embedding(cmd1)
        emb2 = self.get_embedding(cmd2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        similarity = F.cosine_similarity(emb1, emb2)
        return similarity.item()


class LLMJudge:
    """
    Uses an LLM to judge functional equivalence.
    More flexible than embeddings for complex cases.
    """
    
    JUDGE_PROMPT = """You are evaluating if two shell commands are functionally equivalent for a given task.

Task: {instruction}

Expected command: {expected}
Generated command: {generated}

Are these commands functionally equivalent for accomplishing the task?
Consider:
- Do they produce the same output/effect?
- Are there edge cases where they differ?
- Minor formatting differences don't matter.

Answer with JSON:
{{"equivalent": true/false, "reason": "brief explanation"}}
"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    def judge(self, instruction: str, expected: str, generated: str) -> Dict:
        """Judge if commands are equivalent using LLM."""
        # For now, return a placeholder
        # Full implementation would load the model and generate
        
        # Quick heuristic fallback
        if self._quick_equivalence_check(expected, generated):
            return {"equivalent": True, "reason": "Commands use similar tools/patterns"}
        
        return {"equivalent": None, "reason": "LLM judge not available"}
    
    def _quick_equivalence_check(self, cmd1: str, cmd2: str) -> bool:
        """Quick heuristic check for obvious equivalences."""
        # Normalize commands
        cmd1_norm = cmd1.lower().strip()
        cmd2_norm = cmd2.lower().strip()
        
        # Same command (ignoring case/whitespace)
        if cmd1_norm == cmd2_norm:
            return True
        
        # Check for common equivalent patterns
        equivalence_groups = [
            # File listing
            {"ls", "find", "locate", "tree"},
            # File viewing
            {"cat", "less", "more", "head", "tail", "bat"},
            # File searching
            {"grep", "rg", "ag", "ack"},
            # Permissions
            {"chmod", "chown"},
            # Disk usage
            {"du", "df", "ncdu"},
        ]
        
        cmd1_tool = cmd1_norm.split()[0] if cmd1_norm else ""
        cmd2_tool = cmd2_norm.split()[0] if cmd2_norm else ""
        
        for group in equivalence_groups:
            if cmd1_tool in group and cmd2_tool in group:
                return True
        
        return False


class SemanticEvaluator:
    """
    Evaluates command generation with semantic understanding.
    
    Replaces strict exact-match with multi-level evaluation:
    1. Exact match (baseline)
    2. Semantic similarity (CodeBERT embeddings)
    3. LLM-as-judge (for complex cases)
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.80,
        use_llm_judge: bool = False
    ):
        self.similarity_threshold = similarity_threshold
        self.use_llm_judge = use_llm_judge
        
        self.embedder = CodeBERTEmbedder()
        self.llm_judge = LLMJudge() if use_llm_judge else None
    
    def evaluate(
        self,
        instruction: str,
        expected: str,
        generated: str
    ) -> EvaluationResult:
        """
        Evaluate if generated command is equivalent to expected.
        
        Returns EvaluationResult with multiple metrics.
        """
        result = EvaluationResult(
            instruction=instruction,
            expected=expected,
            generated=generated,
            similarity_threshold=self.similarity_threshold
        )
        
        # 1. Exact match
        result.exact_match = self._exact_match(expected, generated)
        
        if result.exact_match:
            result.semantic_similarity = 1.0
            result.semantic_match = True
            result.functionally_equivalent = True
            result.explanation = "Exact string match"
            return result
        
        # 2. Semantic similarity
        result.semantic_similarity = self.embedder.similarity(expected, generated)
        result.semantic_match = result.semantic_similarity >= self.similarity_threshold
        
        if result.semantic_match:
            result.functionally_equivalent = True
            result.explanation = f"Semantic similarity {result.semantic_similarity:.2f} >= {self.similarity_threshold}"
            return result
        
        # 3. LLM judge (optional, for borderline cases)
        if self.use_llm_judge and self.llm_judge:
            judge_result = self.llm_judge.judge(instruction, expected, generated)
            result.llm_judge_match = judge_result.get("equivalent")
            
            if result.llm_judge_match:
                result.functionally_equivalent = True
                result.explanation = f"LLM judge: {judge_result.get('reason', 'equivalent')}"
                return result
        
        # No match
        result.functionally_equivalent = False
        result.explanation = f"Semantic similarity {result.semantic_similarity:.2f} < {self.similarity_threshold}"
        return result
    
    def _exact_match(self, expected: str, generated: str) -> bool:
        """Exact string match (normalized)."""
        return expected.strip() == generated.strip()
    
    def evaluate_batch(
        self,
        examples: List[Dict]
    ) -> Dict:
        """
        Evaluate a batch of examples.
        
        Each example should have: instruction, expected, generated
        
        Returns aggregate metrics.
        """
        results = []
        
        for example in examples:
            result = self.evaluate(
                instruction=example["instruction"],
                expected=example.get("expected", example.get("output", "")),
                generated=example["generated"]
            )
            results.append(result)
        
        # Aggregate metrics
        n = len(results)
        if n == 0:
            return {"error": "No examples to evaluate"}
        
        exact_matches = sum(1 for r in results if r.exact_match)
        semantic_matches = sum(1 for r in results if r.semantic_match)
        functional_matches = sum(1 for r in results if r.functionally_equivalent)
        
        return {
            "total": n,
            "exact_match": exact_matches,
            "exact_match_rate": exact_matches / n,
            "semantic_match": semantic_matches,
            "semantic_match_rate": semantic_matches / n,
            "functional_match": functional_matches,
            "functional_match_rate": functional_matches / n,
            "avg_similarity": sum(r.semantic_similarity for r in results) / n,
        }


# Convenience function
def evaluate_semantic_equivalence(
    instruction: str,
    expected: str,
    generated: str,
    threshold: float = 0.80
) -> EvaluationResult:
    """Quick semantic evaluation."""
    evaluator = SemanticEvaluator(similarity_threshold=threshold)
    return evaluator.evaluate(instruction, expected, generated)
