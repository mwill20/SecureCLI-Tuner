---
description: Security regression testing requirements for SecureCLI-Tuner
---

# Security Regression Standard

> **Mandate:** Every model update must pass automated adversarial testing before merge to main.

## Requirements

### Adversarial Pass Rate Threshold

| Metric | Minimum | Target |
|--------|---------|--------|
| **Adversarial Pass Rate** | **>95%** | 99% |
| Direct Attack Detection | >98% | 100% |
| Obfuscated Attack Detection | >90% | 95% |
| Prompt Injection Resistance | >95% | 99% |

### Enforcement

1. **Pre-Commit Hook**
   - Run `tests/eval/test_adversarial.py` before every commit
   - Block commit if pass rate < 95%

2. **CI/CD Pipeline**
   - Mandatory adversarial regression on every PR to `main`
   - W&B logging to `SecureCLI-Monitoring` project
   - Auto-reject PRs below threshold

3. **Model Update Protocol**
   - Any change to `training/` requires full adversarial suite run
   - Any change to `commandrisk/` requires full adversarial suite run
   - Results must be documented in PR description

## Adversarial Suite Categories

| Category | Test Count | Layer Coverage |
|----------|------------|----------------|
| Direct Destructive | 15+ | Layer 1 |
| Obfuscated (base64, hex) | 10+ | Layer 3 |
| Privilege Escalation | 10+ | Layer 1, 2 |
| Data Exfiltration | 10+ | Layer 2 |
| Prompt Injection | 20+ | Layer 3 |
| Goal Hijacking | 10+ | Semantic Guardrail |

## Execution

```powershell
# Run full adversarial regression
python -m pytest tests/eval/test_adversarial.py -v --tb=short

# Generate regression report
python evaluation/generate_eval_report.py --adversarial

# Check pass rate
python evaluation/check_threshold.py --min-rate 0.95
```

## W&B Tracking

Every regression run logs to `SecureCLI-Monitoring`:

- `adversarial_pass_rate` — Primary metric
- `layer1_catch_rate` — Deterministic coverage
- `layer2_catch_rate` — Heuristic coverage  
- `layer3_catch_rate` — Semantic coverage
- `regression_timestamp` — Run timestamp
- `commit_hash` — Associated git commit

## Failure Protocol

If adversarial pass rate drops below 95%:

1. **Block the merge** — No exceptions
2. **Root cause analysis** — Identify which patterns failed
3. **Update guardrails** — Add missing patterns or improve semantic model
4. **Re-run full suite** — Must pass before proceeding
5. **Document in SECURITY.md** — Record the incident and fix

## OWASP Compliance Note

> SecureCLI-Tuner adheres to the OWASP Top 10 for Agentic Applications (2026).
> Every CommandRisk block is cross-referenced with ASI and MITRE ATT&CK identifiers
> for professional security reporting.
