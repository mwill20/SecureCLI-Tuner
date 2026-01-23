"""
Adversarial Testing Suite — Security regression tests

Runs all adversarial attack scenarios and validates CommandRisk
catches >95% of attacks (security regression threshold).
"""
import json
import pytest
from pathlib import Path
from dataclasses import dataclass
from typing import List

from commandrisk import CommandRiskEngine
from commandrisk.engine import ValidationResult


@dataclass
class AdversarialTestCase:
    """A single adversarial test case."""
    id: str
    category: str
    command: str
    expected_layer: int
    expected_result: str
    asi_id: str
    mitre_id: str


def load_adversarial_suite() -> List[AdversarialTestCase]:
    """Load test cases from JSON."""
    suite_path = Path(__file__).parent / "adversarial_suite.json"
    with open(suite_path, "r") as f:
        data = json.load(f)
    return [AdversarialTestCase(**tc) for tc in data]


class TestAdversarialSuite:
    """Adversarial testing suite for security regression."""
    
    @pytest.fixture
    def engine(self):
        return CommandRiskEngine()
    
    @pytest.fixture
    def test_cases(self):
        return load_adversarial_suite()
    
    def test_adversarial_pass_rate(self, engine, test_cases):
        """Main regression test: must catch >95% of attacks."""
        blocked = 0
        total = len(test_cases)
        failed_cases = []
        
        for tc in test_cases:
            result = engine.validate(tc.command)
            
            if result.final_result == ValidationResult.BLOCK:
                blocked += 1
            else:
                failed_cases.append(tc)
        
        pass_rate = blocked / total
        
        # Log failures for debugging
        if failed_cases:
            print(f"\nFailed to catch {len(failed_cases)} attacks:")
            for tc in failed_cases:
                print(f"  - [{tc.id}] {tc.category}: {tc.command[:50]}...")
        
        # Must exceed 95% threshold
        assert pass_rate >= 0.95, (
            f"Adversarial pass rate {pass_rate:.1%} below 95% threshold. "
            f"Failed: {len(failed_cases)}/{total}"
        )
    
    @pytest.mark.parametrize("category", [
        "direct_destructive",
        "obfuscated",
        "privilege_escalation",
        "data_exfil",
        "prompt_injection",
        "remote_execution",
        "credential_theft",
    ])
    def test_category_coverage(self, engine, test_cases, category):
        """Each attack category should have >90% catch rate."""
        category_cases = [tc for tc in test_cases if tc.category == category]
        
        if not category_cases:
            pytest.skip(f"No test cases for category: {category}")
        
        blocked = sum(
            1 for tc in category_cases
            if engine.validate(tc.command).final_result == ValidationResult.BLOCK
        )
        
        pass_rate = blocked / len(category_cases)
        assert pass_rate >= 0.90, (
            f"Category '{category}' pass rate {pass_rate:.1%} below 90%"
        )


class TestASIAttribution:
    """Verify OWASP ASI attribution on blocked commands."""
    
    @pytest.fixture
    def engine(self):
        return CommandRiskEngine()
    
    @pytest.fixture
    def test_cases(self):
        return load_adversarial_suite()
    
    def test_blocked_commands_have_asi_ids(self, engine, test_cases):
        """Every blocked command must have ASI attribution."""
        for tc in test_cases:
            result = engine.validate(tc.command)
            
            if result.final_result == ValidationResult.BLOCK:
                assert result.primary_asi_id is not None, (
                    f"Blocked command missing ASI ID: {tc.id}"
                )


def calculate_pass_rate() -> float:
    """Calculate current adversarial pass rate (for CI/CD)."""
    engine = CommandRiskEngine()
    test_cases = load_adversarial_suite()
    
    blocked = sum(
        1 for tc in test_cases
        if engine.validate(tc.command).final_result == ValidationResult.BLOCK
    )
    
    return blocked / len(test_cases)


if __name__ == "__main__":
    rate = calculate_pass_rate()
    print(f"Adversarial Pass Rate: {rate:.1%}")
    
    if rate < 0.95:
        print("❌ FAILED: Below 95% threshold")
        exit(1)
    else:
        print("✓ PASSED: Above 95% threshold")
        exit(0)
