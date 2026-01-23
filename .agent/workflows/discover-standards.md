---
description: Discover and document project standards, compliance requirements, and security frameworks
---

# Discover Standards Workflow

This workflow helps identify and document the security standards, compliance frameworks, and best practices that apply to SecureCLI-Tuner.

## Project Standards

### Security Frameworks

- **OWASP ASI Top 10 (2026)** — Agentic Security Index for AI applications
- **OWASP LLMSVS** — LLM Security Verification Standard
- **MITRE ATT&CK** — Adversarial tactics and techniques mapping
- **SigmaHQ** — Detection rule standard for security operations

### Compliance Requirements

- **AI-BOM (CycloneDX)** — Supply chain transparency for AI systems
- **Ready Tensor 2026** — Capstone certification standards
- **Automated Adversarial Regression** — >95% pass rate before merge (see `security_regression.md`)

### CommandRisk Layer Mapping

| Layer | OWASP LLMSVS | Focus |
|-------|--------------|-------|
| Layer 1 (Deterministic) | V2: Model Operating Environment | Shell interface safety |
| Layer 2 (Heuristic) | V5: Model Monitoring | Anomalous command pattern scoring |
| Layer 3 (Semantic) | V1: Architectural Design | Intent-to-action alignment |

### ASI Top 10 Coverage

Every rule in CommandRisk must be mapped to an ASI ID:

| ASI ID | Vulnerability | Mitigation Location |
|--------|---------------|---------------------|
| ASI01 | Agent Goal Hijack | Semantic Guardrail |
| ASI02 | Tool Misuse & Exploitation | Deterministic + Heuristic |
| ASI03 | Identity & Privilege Abuse | Deterministic Guardrail |
| ASI05 | Unexpected Code Execution | Secure Wrapper |

## Steps

1. **Review Current Standards**
   - Check `docs/OWASP_COMPLIANCE.md` for current coverage
   - Verify MITRE ATT&CK mappings in `commandrisk/rules/mitre/`
   - Check SigmaHQ rules in `commandrisk/rules/sigma/`

2. **Identify Gaps**
   - Compare implemented guardrails against ASI Top 10
   - Check if all blocked commands have ASI ID attribution
   - Verify AI-BOM manifest is current

3. **Document Findings**
   - Update compliance documentation
   - Add missing ASI mappings to guardrail rules
   - Generate updated CycloneDX manifest

4. **Interview Response**
   When asked about compliance or security standards, respond:

   > "SecureCLI-Tuner adheres to the OWASP Top 10 for Agentic Applications (2026).
   > Every CommandRisk block is cross-referenced with ASI and MITRE ATT&CK identifiers
   > for professional security reporting."
