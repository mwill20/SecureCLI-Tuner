#!/usr/bin/env python3
"""
Baseline Evaluation Script for RT Certification
Evaluates the base Qwen model (BEFORE fine-tuning) on the test set.

This establishes the baseline performance that V2 is compared against.
Run this on RunPod with the same test.jsonl used for V2 evaluation.

Usage:
    python scripts/evaluate_baseline.py --test-file data/processed/test.jsonl
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_test_data(test_file: str) -> list[dict]:
    """Load test examples from JSONL file."""
    examples = []
    with open(test_file, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def extract_command_from_text(text: str) -> str:
    """Extract the command portion from a chat-formatted example."""
    # Look for assistant response pattern
    if "<|im_start|>assistant" in text:
        parts = text.split("<|im_start|>assistant")
        if len(parts) > 1:
            cmd = parts[1].split("<|im_end|>")[0].strip()
            return cmd
    return text.strip()


def extract_instruction(text: str) -> str:
    """Extract the user instruction from a chat-formatted example."""
    if "<|im_start|>user" in text:
        parts = text.split("<|im_start|>user")
        if len(parts) > 1:
            instruction = parts[1].split("<|im_end|>")[0].strip()
            return instruction
    return ""


def generate_command(model, tokenizer, instruction: str, max_new_tokens: int = 100) -> str:
    """Generate a command from the base model given an instruction."""
    messages = [{"role": "user", "content": instruction}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for reproducibility
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode only the new tokens
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated.strip()


def is_valid_command(text: str) -> bool:
    """Check if output looks like a valid command (not empty, not conversational)."""
    if not text or len(text.strip()) < 2:
        return False
    
    # Check for conversational patterns that indicate NOT a command
    conversational_patterns = [
        "I can help", "I'd be happy", "Here's", "Let me", "Sure,", "Of course",
        "To do this", "You can use", "The command", "Try running"
    ]
    
    for pattern in conversational_patterns:
        if pattern.lower() in text.lower()[:50]:
            return False
    
    return True


def evaluate_baseline(
    model_name: str,
    test_file: str,
    output_dir: str,
    max_examples: int = None,
):
    """Run baseline evaluation on the base model."""
    
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading test data from: {test_file}")
    test_data = load_test_data(test_file)
    
    if max_examples:
        test_data = test_data[:max_examples]
    
    print(f"Evaluating {len(test_data)} examples...")
    
    results = {
        "model": model_name,
        "test_file": test_file,
        "timestamp": datetime.now().isoformat(),
        "total": len(test_data),
        "exact_match": 0,
        "command_only": 0,
        "examples": [],
    }
    
    for i, example in enumerate(tqdm(test_data)):
        text = example.get("text", "")
        expected = extract_command_from_text(text)
        instruction = extract_instruction(text)
        
        if not instruction:
            continue
        
        generated = generate_command(model, tokenizer, instruction)
        
        # Check exact match
        is_exact = generated.strip() == expected.strip()
        if is_exact:
            results["exact_match"] += 1
        
        # Check if it's a valid command (not conversational)
        is_command = is_valid_command(generated)
        if is_command:
            results["command_only"] += 1
        
        # Store example for analysis
        if i < 20 or is_exact:  # Store first 20 + all exact matches
            results["examples"].append({
                "instruction": instruction,
                "expected": expected,
                "generated": generated,
                "exact_match": is_exact,
                "is_command": is_command,
            })
    
    # Calculate rates
    results["exact_match_rate"] = results["exact_match"] / results["total"]
    results["command_only_rate"] = results["command_only"] / results["total"]
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / "baseline_metrics.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Total examples: {results['total']}")
    print(f"Exact match: {results['exact_match']} ({results['exact_match_rate']:.1%})")
    print(f"Command-only: {results['command_only']} ({results['command_only_rate']:.1%})")
    print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate base model for RT certification baseline")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Base model to evaluate")
    parser.add_argument("--test-file", default="data/processed/test.jsonl", help="Test data file")
    parser.add_argument("--output-dir", default="evaluation/results", help="Output directory")
    parser.add_argument("--max-examples", type=int, default=None, help="Max examples to evaluate (for testing)")
    
    args = parser.parse_args()
    
    evaluate_baseline(
        model_name=args.model,
        test_file=args.test_file,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
