# Lesson 3: Safety Evaluation

## 1. Introduction

### Learning Objectives

- Run domain and safety evaluation suites
- Understand exact match vs command-only metrics
- Test adversarial robustness
- Achieve >95% adversarial pass rate

> [!NOTE]
> **VERIFIED DATA:** V2 achieved **100% adversarial pass rate** (9/9 tests) and **9.1% exact match** in production run `honest-music-2` (2026-01-24).

### Why Rigorous Evaluation?

V1 showed 57% adversarial safe rate—43% of attack prompts bypassed training-time safety. Evaluation reveals these gaps **before deployment**.

---

## 2. Evaluation Metrics

### Domain Metrics

| Metric | Description | V1 Result |
|--------|-------------|-----------|
| **Exact Match** | Strict string comparison | 13.22% |
| **Command-Only Rate** | Model outputs just commands | 99.4% |
| **Syntax Validity** | Shellcheck validation | 97% |

### Safety Metrics

| Metric | Description | V1 Result | V2 Result |
|--------|-------------|-----------|-----------|
| **Test Set Safe** | 0 dangerous in test outputs | PASS | ✅ PASS |
| **Adversarial Safe** | Attack prompts blocked | 57% | **100%** |

---

## 3. Running Evaluation

### Domain Evaluation

```powershell
python scripts/evaluate_semantic.py `
  --checkpoint models/checkpoints/checkpoint-500 `
  --test-data data/processed/test.jsonl `
  --output-dir evaluation/semantic
```

### Safety Evaluation

```powershell
python evaluation/evaluate_safety.py `
  --checkpoint models/checkpoints/phase2-final `
  --adversarial-prompts data/adversarial/adversarial_prompts.jsonl `
  --output-dir evaluation/safety
```

### Adversarial Suite (with CommandRisk)

```powershell
python -m pytest tests/eval/test_adversarial.py -v
```

**Verified Result:** 100% pass rate (9/9 tests passed)

---

## 4. Adversarial Categories

| Category | Test Count | Expected Result |
|----------|------------|-----------------|
| Root Deletion | 4 | Layer 1 BLOCK |
| Fork Bomb | 2 | Layer 1 BLOCK |
| Disk Wipe | 2 | Layer 1 BLOCK |
| Permission Bomb | 2 | Layer 1 BLOCK |
| Remote Execution | 4 | Layer 1 BLOCK |
| Prompt Injection | 3 | Layer 3 BLOCK |
| Privilege Escalation | 1 | Layer 1 BLOCK |

---

## 5. Interview Preparation

### Q: What's the difference between exact match and command-only rate?

**Model Answer:** "Exact match is strict string comparison—commands must match character-for-character. It's a conservative metric; `ls -la` vs `ls -al` counts as a miss even though they're equivalent. Command-only rate measures whether the model outputs just a command (not explanations or markdown). V1 achieved 99.4% command-only, showing the model learned the task format."

### Q: Why did V1 only achieve 57% adversarial safe rate?

**Model Answer:** "Training-time filtering removes dangerous examples from training data, but the model can still generate dangerous commands from adversarial prompts. V1 had no runtime guardrails. V2 adds CommandRisk—a 3-layer security engine that catches attacks at inference time."

---

## 6. Key Takeaways

- ✅ Domain metrics validate task learning
- ✅ Adversarial testing reveals safety gaps
- ✅ **V2 achieved 100% adversarial pass rate** (9/9 tests)
- ✅ CommandRisk runtime guardrails are essential

---

## 7. Next Steps

- Proceed to [Lesson 4: CommandRisk Engine](Lesson_04_CommandRisk.md)
