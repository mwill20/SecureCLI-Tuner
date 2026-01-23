# Security Design

## Overview

SecureCLI-Tuner implements a **Zero-Trust Security Kernel** that validates every generated command before execution.

## Architecture

```
User Query → Router → Generator (LLM) → CommandRisk Engine → Secure Wrapper → Execution
                                              ↓
                              ┌───────────────┼───────────────┐
                              ↓               ↓               ↓
                        Deterministic    Semantic       Policy
                        Guardrail        Guardrail      Guardrail
                              ↓               ↓               ↓
                         SigmaHQ         Intent ML       OWASP ASI
```

## Guardrail Layers

### Layer 1: Deterministic

- **Purpose**: Hard blocks via regex patterns
- **Coverage**: 17+ zero-tolerance patterns
- **LLMSVS**: V2 (Model Operating Environment)
- **Response Time**: <1ms

### Layer 2: Heuristic

- **Purpose**: Risk scoring
- **Method**: Command complexity + target sensitivity + MITRE mapping
- **Threshold**: Score >70 triggers block
- **LLMSVS**: V5 (Model Monitoring)

### Layer 3: Semantic

- **Purpose**: Intent-to-action alignment
- **Method**: ML classifier + obfuscation detection
- **Coverage**: Base64, hex, eval, nested substitution
- **LLMSVS**: V1 (Architectural Design)

## Trust Model

1. **User input is untrusted** — May contain prompt injection
2. **LLM output is untrusted** — May hallucinate dangerous commands
3. **Only validated commands pass** — After 3-layer inspection

## Threat Mitigations

| Threat | Mitigation |
|--------|------------|
| Destructive commands | Deterministic blocklist |
| Privilege escalation | Pattern + policy rules |
| Data exfiltration | Heuristic scoring |
| Prompt injection | Semantic detection |
| Goal hijacking | Intent alignment |
| Obfuscated attacks | Base64/hex/eval detection |

## Incident Response

All blocked commands are logged with:

- Timestamp
- Original command
- Blocking guardrail
- ASI ID attribution
- MITRE ATT&CK technique
- Risk score

Logs are sent to W&B `SecureCLI-Monitoring` for real-time alerting.
