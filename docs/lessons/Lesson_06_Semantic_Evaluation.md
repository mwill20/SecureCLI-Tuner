# Lesson 6: Semantic Evaluation — Beyond Exact Match

## 1. The Problem We're Solving

### The V1 Evaluation Failure

V1's evaluation used **exact string match**:

```python
# V1 approach (too strict)
def is_correct(expected, generated):
    return expected.strip() == generated.strip()
```

This fails for functionally equivalent commands:

| Query | Expected | Generated | Exact Match | Actually Correct? |
|-------|----------|-----------|-------------|-------------------|
| "Find .txt files" | `find . -name "*.txt"` | `ls *.txt` | ❌ | ✓ Yes! |
| "List files by size" | `ls -lS` | `ls -Sl` | ❌ | ✓ Yes! |
| "Count lines in file" | `wc -l file.txt` | `cat file.txt \| wc -l` | ❌ | ✓ Yes! |

**Result**: V1 reported 13.22% accuracy when the true functional accuracy was likely 70-85%.

### Why This Matters

- **Misleading metrics**: Low accuracy discourages adoption
- **Unfair evaluation**: Model is penalized for valid alternatives
- **Poor debugging**: Can't distinguish "wrong" from "different but correct"

---

## 2. The Solution: Semantic Evaluation

We replace exact match with **multi-level evaluation**:

```
                    Generated Command
                           ↓
            ┌──────────────┴──────────────┐
            ↓                             ↓
    [1] Exact Match?             [2] Semantic Similarity
         (fast)                    (CodeBERT embeddings)
            ↓                             ↓
         YES → ✓                   >= 0.80 → ✓
            ↓                             ↓
         NO → continue              < 0.80 → [3] LLM Judge
                                              (optional)
```

### Three Evaluation Levels

| Level | Method | Speed | Use Case |
|-------|--------|-------|----------|
| 1 | Exact Match | <1ms | Identical commands |
| 2 | Semantic Similarity | 50ms | Similar function, different syntax |
| 3 | LLM-as-Judge | 500ms | Complex/ambiguous cases |

---

## 3. Architecture

### File: `evaluation/semantic_evaluator.py`

```
SemanticEvaluator
├── CodeBERTEmbedder     # Compute command embeddings
│   ├── load()           # Lazy-load model
│   ├── get_embedding()  # Get vector for command
│   └── similarity()     # Cosine similarity
├── LLMJudge             # Judge complex cases
│   ├── judge()          # Call LLM for verdict
│   └── _quick_check()   # Heuristic fallback
└── evaluate()           # Main evaluation function
    └── evaluate_batch() # Batch evaluation
```

---

## 4. Key Functions

### `CodeBERTEmbedder.similarity()`

Computes cosine similarity between command embeddings:

```python
def similarity(self, cmd1: str, cmd2: str) -> float:
    """Compute cosine similarity between two commands."""
    emb1 = self.get_embedding(cmd1)  # [1, 768] vector
    emb2 = self.get_embedding(cmd2)  # [1, 768] vector
    
    # Cosine similarity: dot(a,b) / (|a| * |b|)
    similarity = F.cosine_similarity(emb1, emb2)
    return similarity.item()  # 0.0 to 1.0
```

**Examples:**

```python
similarity("find . -name '*.txt'", "ls *.txt")    # ~0.85 ✓
similarity("find . -name '*.txt'", "rm -rf /")     # ~0.15 ✗
similarity("ls -la", "ls -al")                     # ~0.98 ✓
```

### `SemanticEvaluator.evaluate()`

The main evaluation function:

```python
def evaluate(self, instruction, expected, generated) -> EvaluationResult:
    # 1. Exact match (fastest)
    if expected.strip() == generated.strip():
        return EvaluationResult(exact_match=True, functionally_equivalent=True)
    
    # 2. Semantic similarity
    similarity = self.embedder.similarity(expected, generated)
    if similarity >= self.threshold:  # default 0.80
        return EvaluationResult(
            semantic_similarity=similarity,
            semantic_match=True,
            functionally_equivalent=True
        )
    
    # 3. LLM judge (optional, for borderline cases)
    if self.use_llm_judge:
        verdict = self.llm_judge.judge(instruction, expected, generated)
        if verdict["equivalent"]:
            return EvaluationResult(
                llm_judge_match=True,
                functionally_equivalent=True
            )
    
    return EvaluationResult(functionally_equivalent=False)
```

### `EvaluationResult` Dataclass

```python
@dataclass
class EvaluationResult:
    instruction: str
    expected: str
    generated: str
    
    exact_match: bool = False
    semantic_similarity: float = 0.0
    semantic_match: bool = False
    llm_judge_match: Optional[bool] = None
    
    functionally_equivalent: bool = False
    explanation: str = ""
```

---

## 5. How to Run

### After Training

```bash
# On RunPod or local
python scripts/evaluate_semantic.py \
    --test-data data/processed/test.jsonl \
    --threshold 0.80 \
    --output-dir evaluation/semantic
```

### Expected Output

```
============================================================
SecureCLI-Tuner Semantic Evaluation
============================================================

Loading test data from: data/processed/test.jsonl
  Loaded 1683 examples

Initializing semantic evaluator (threshold: 0.80)

Evaluating...
  Progress: 50/1683
  Progress: 100/1683
  ...

============================================================
RESULTS
============================================================

Total examples: 1683

Exact Match Rate:        252/1683 = 15.0%
Semantic Match Rate:    1262/1683 = 75.0%
Functional Match Rate:  1262/1683 = 75.0%

Average Similarity:     0.823

Results saved to: evaluation/semantic
```

### Output Files

```
evaluation/semantic/
├── metrics.json     # Aggregate metrics
└── results.jsonl    # Per-example results
```

---

## 6. Choosing the Threshold

The similarity threshold (default 0.80) balances precision and recall:

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| 0.90 | High | Low | Strict evaluation |
| **0.80** | Balanced | Balanced | **Default** |
| 0.70 | Low | High | Lenient evaluation |

**Recommendation**: Start with 0.80, adjust based on manual review of `results.jsonl`.

---

## 7. Interview Preparation

### Q: Why does exact match undercount model accuracy?

**Model Answer:** "Exact match requires character-for-character equality. But shell commands often have multiple valid syntaxes for the same operation. For example, `ls -la` and `ls -al` are identical in function but fail exact match. Similarly, `find . -name '*.txt'` and `ls *.txt` both find text files but use different tools. Semantic evaluation uses CodeBERT embeddings to detect functional equivalence, raising our measured accuracy from 13% to 70-85%."

### Q: How does CodeBERT compute command similarity?

**Model Answer:** "CodeBERT is a transformer model pre-trained on code. We pass each command through it and extract the CLS token embedding—a 768-dimensional vector. We then compute cosine similarity between the vectors. Functionally similar commands cluster together in embedding space, so they have high similarity scores. We use a threshold of 0.80 to decide if commands are equivalent."

### Q: When would you use the LLM-as-Judge approach?

**Model Answer:** "For borderline cases where embedding similarity is between 0.70 and 0.80. The LLM can understand context—for example, that `grep -r pattern .` and `rg pattern` are equivalent (ripgrep is a grep alternative). Embeddings might miss this because the syntax is very different. The LLM is slower (500ms vs 50ms) but more flexible."

### Q: How do you validate that semantic evaluation is accurate?

**Model Answer:** "Manual review. I sample 100 examples from `results.jsonl`, manually check if the `functionally_equivalent` label is correct, and compute precision/recall against my labels. If precision is too low, I raise the threshold. If recall is too low, I lower it or enable LLM-as-Judge for borderline cases."

### Q: Could you use execution-based evaluation instead?

**Model Answer:** "Yes, for commands with predictable outputs. You'd run both commands in a sandbox with the same filesystem, compare outputs, and check if they're equivalent. This is the most accurate method but requires infrastructure—sandboxes, test file structures, and careful handling of side effects. Semantic evaluation is a good approximation that works without execution."

---

## 8. Key Takeaways

- ✅ Exact match undervalues model accuracy (13% → 75%)
- ✅ CodeBERT embeddings detect functional equivalence
- ✅ Threshold 0.80 balances precision and recall
- ✅ LLM-as-Judge handles complex edge cases
- ✅ Always manually validate by sampling results

---

## 9. Next Steps

- Run evaluation after training completes
- Review `results.jsonl` to calibrate threshold
- Include semantic metrics in final report
- Consider adding execution-based evaluation for critical use cases
