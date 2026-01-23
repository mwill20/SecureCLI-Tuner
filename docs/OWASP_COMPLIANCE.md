# OWASP Compliance Documentation

> SecureCLI-Tuner adheres to the OWASP Top 10 for Agentic Applications (2026).
> Every CommandRisk block is cross-referenced with ASI and MITRE ATT&CK identifiers
> for professional security reporting.

## ASI Top 10 Coverage

| ASI ID | Vulnerability | Status | Implementation |
|--------|---------------|--------|----------------|
| ASI01 | Agent Goal Hijack | ✅ Covered | Semantic Guardrail |
| ASI02 | Tool Misuse & Exploitation | ✅ Covered | Deterministic + Heuristic |
| ASI03 | Identity & Privilege Abuse | ✅ Covered | Deterministic Guardrail |
| ASI04 | Insufficient Sandboxing | 🔄 Partial | Secure Wrapper |
| ASI05 | Unexpected Code Execution | ✅ Covered | Secure Wrapper |
| ASI06 | Poor Secret Management | 🔄 Partial | Credential detection |
| ASI07 | Lack of Observability | ✅ Covered | W&B integration |
| ASI08 | Insufficient Logging | ✅ Covered | Audit log |
| ASI09 | Unsafe Resource Access | 🔄 Partial | Policy Guardrail |
| ASI10 | Unvalidated Tool Outputs | ✅ Covered | CommandRisk engine |

## LLMSVS Mapping

| LLMSVS Category | Layer Coverage |
|-----------------|----------------|
| V1: Architectural Design | Layer 3 (Semantic) |
| V2: Model Operating Environment | Layer 1 (Deterministic) |
| V3: Model Security | Training pipeline |
| V4: Model Deployment | Secure Wrapper |
| V5: Model Monitoring | Layer 2 (Heuristic) + W&B |

## Compliance Artifacts

- `/compliance/cyclonedx/` — AI-BOM manifests
- `/commandrisk/rules/owasp/` — ASI rule definitions
- `/tests/eval/adversarial_suite.json` — ASI test coverage

## Verification

```powershell
# Run OWASP compliance tests
python -m pytest tests/eval/test_adversarial.py -v

# Generate compliance report
python compliance/generate_report.py
```
