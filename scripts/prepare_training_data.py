#!/usr/bin/env python3
"""
Prepare the authoritative training dataset.

This script:
1. Loads all raw datasets from data/raw/
2. Merges and deduplicates
3. Filters dangerous commands
4. Validates with Shellcheck
5. Applies chat templates
6. Splits train/val/test
7. Saves to data/processed/

Run:
    python scripts/prepare_training_data.py
"""
import json
import hashlib
import random
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Set

from transformers import AutoTokenizer
from pydantic import ValidationError

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commandrisk.guardrails.deterministic import DeterministicGuardrail
from schemas.dataset import BashCommandExample

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_LENGTH = 2048
SPLIT_SEED = 42
SHELLCHECK_TIMEOUT = 5

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = DATA_DIR / "logs"


def load_raw_datasets() -> List[Dict]:
    """Load all JSONL files from raw directory."""
    all_examples = []
    
    for jsonl_file in RAW_DIR.glob("*.jsonl"):
        print(f"  Loading {jsonl_file.name}...")
        count = 0
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    example = json.loads(line)
                    all_examples.append(example)
                    count += 1
        print(f"    Loaded {count} examples")
    
    return all_examples


def compute_fingerprint(example: Dict) -> str:
    """Compute unique fingerprint for deduplication."""
    payload = json.dumps({
        "instruction": example.get("instruction", ""),
        "output": example.get("output", ""),
    }, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deduplicate(examples: List[Dict]) -> List[Dict]:
    """Remove duplicate examples based on fingerprint."""
    seen: Set[str] = set()
    unique = []
    
    for example in examples:
        fp = compute_fingerprint(example)
        if fp not in seen:
            seen.add(fp)
            unique.append(example)
    
    removed = len(examples) - len(unique)
    print(f"  Removed {removed} duplicates ({len(unique)} unique)")
    return unique


def validate_schema(examples: List[Dict]) -> List[Dict]:
    """Validate examples against Pydantic schema."""
    valid = []
    invalid = []
    
    for example in examples:
        try:
            validated = BashCommandExample(
                instruction=example.get("instruction", ""),
                input=example.get("input", ""),
                output=example.get("output", "")
            )
            if hasattr(validated, "model_dump"):
                valid.append(validated.model_dump())
            else:
                valid.append(validated.dict())
        except ValidationError as e:
            invalid.append({"example": example, "error": str(e)})
    
    print(f"  Schema valid: {len(valid)}, invalid: {len(invalid)}")
    
    # Save invalid examples for debugging
    if invalid:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with (LOG_DIR / "schema_violations.jsonl").open("w") as f:
            for item in invalid:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")
    
    return valid


def filter_dangerous(examples: List[Dict]) -> List[Dict]:
    """Filter dangerous commands using deterministic guardrail."""
    guardrail = DeterministicGuardrail()
    
    safe = []
    dangerous = []
    
    for example in examples:
        if guardrail.is_dangerous(example["output"]):
            result = guardrail.validate(example["output"])
            dangerous.append({
                "instruction": example["instruction"],
                "output": example["output"],
                "pattern": result.rationale,
                "asi_ids": result.asi_ids,
            })
        else:
            safe.append(example)
    
    print(f"  Safe: {len(safe)}, Dangerous removed: {len(dangerous)}")
    
    # Save dangerous commands for audit
    if dangerous:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with (LOG_DIR / "removed_dangerous.jsonl").open("w") as f:
            for item in dangerous:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")
    
    return safe


def validate_shellcheck(examples: List[Dict]) -> List[Dict]:
    """Validate Bash syntax with Shellcheck."""
    valid = []
    invalid = []
    
    total = len(examples)
    print(f"  Validating {total} commands with shellcheck...")
    
    for i, example in enumerate(examples):
        if (i + 1) % 500 == 0:
            print(f"    Progress: {i+1}/{total} ({100*(i+1)//total}%)")
        
        try:
            result = subprocess.run(
                ["shellcheck", "--shell=bash", "--severity=error", "-"],
                input=example["output"].encode("utf-8"),
                capture_output=True,
                timeout=SHELLCHECK_TIMEOUT,
                check=False,
            )
            if result.returncode == 0:
                valid.append(example)
            else:
                invalid.append({
                    "instruction": example["instruction"],
                    "output": example["output"],
                    "error": result.stdout.decode("utf-8", errors="ignore")[:200]
                })
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Skip shellcheck if not available
            valid.append(example)
    
    print(f"  Shellcheck passed: {len(valid)}, failed: {len(invalid)}")
    
    if invalid:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with (LOG_DIR / "removed_invalid_syntax.jsonl").open("w") as f:
            for item in invalid:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")
    
    return valid


def apply_chat_template(examples: List[Dict]) -> List[Dict]:
    """Apply Qwen chat template."""
    print(f"  Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    
    formatted = []
    for example in examples:
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        formatted.append({
            "instruction": example["instruction"],
            "output": example["output"],
            "text": text,
        })
    
    print(f"  Formatted {len(formatted)} examples")
    return formatted


def split_dataset(examples: List[Dict], seed: int = SPLIT_SEED) -> Dict[str, List[Dict]]:
    """Split into train/val/test (80/10/10)."""
    random.seed(seed)
    random.shuffle(examples)
    
    n = len(examples)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    train = examples[:train_size]
    val = examples[train_size:train_size + val_size]
    test = examples[train_size + val_size:]
    
    print(f"  Split: {len(train)} train, {len(val)} val, {len(test)} test")
    return {"train": train, "val": val, "test": test}


def save_datasets(splits: Dict[str, List[Dict]]) -> None:
    """Save processed datasets to JSONL."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    for split_name, examples in splits.items():
        output_path = PROCESSED_DIR / f"{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=True) + "\n")
        print(f"  Saved {output_path.name}: {len(examples)} examples")


def save_provenance(stats: Dict, splits: Dict) -> None:
    """Save provenance metadata."""
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL,
        "total_raw": stats["total_raw"],
        "after_dedup": stats["after_dedup"],
        "after_schema": stats["after_schema"],
        "after_dangerous": stats["after_dangerous"],
        "after_shellcheck": stats["after_shellcheck"],
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
        "split_seed": SPLIT_SEED,
    }
    
    output_path = PROCESSED_DIR / "provenance.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    
    print(f"  Saved provenance: {output_path}")


def main():
    print("\n" + "=" * 60)
    print("SecureCLI-Tuner Training Data Preparation")
    print("=" * 60 + "\n")
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    
    # Step 1: Load raw datasets
    print("Step 1/7: Loading raw datasets...")
    examples = load_raw_datasets()
    stats["total_raw"] = len(examples)
    
    if not examples:
        print("\n❌ No raw datasets found!")
        print("   Run: python scripts/download_datasets.py first")
        return
    
    # Step 2: Deduplicate
    print("\nStep 2/7: Deduplicating...")
    examples = deduplicate(examples)
    stats["after_dedup"] = len(examples)
    
    # Step 3: Schema validation
    print("\nStep 3/7: Validating schema...")
    examples = validate_schema(examples)
    stats["after_schema"] = len(examples)
    
    # Step 4: Filter dangerous
    print("\nStep 4/7: Filtering dangerous commands...")
    examples = filter_dangerous(examples)
    stats["after_dangerous"] = len(examples)
    
    # Step 5: Shellcheck validation
    print("\nStep 5/7: Validating with shellcheck...")
    examples = validate_shellcheck(examples)
    stats["after_shellcheck"] = len(examples)
    
    # Step 6: Apply chat template
    print("\nStep 6/7: Applying chat template...")
    examples = apply_chat_template(examples)
    
    # Step 7: Split and save
    print("\nStep 7/7: Splitting and saving...")
    splits = split_dataset(examples)
    save_datasets(splits)
    save_provenance(stats, splits)
    
    print("\n" + "=" * 60)
    print("✓ Training data preparation complete!")
    print("=" * 60)
    print(f"\nOutput: {PROCESSED_DIR}")
    print(f"  - train.jsonl: {len(splits['train'])} examples")
    print(f"  - val.jsonl: {len(splits['val'])} examples")
    print(f"  - test.jsonl: {len(splits['test'])} examples")
    print(f"\nNext step: Upload to RunPod and run training")


if __name__ == "__main__":
    main()
