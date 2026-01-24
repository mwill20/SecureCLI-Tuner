#!/usr/bin/env python3
"""
MMLU Subset Evaluation for RT Certification
Evaluates both base and fine-tuned models on MMLU subset to check for catastrophic forgetting.

This satisfies the RT requirement:
"Include at least one general benchmark (e.g., MMLU subset, HellaSwag, or GSM8K) 
to check for catastrophic forgetting."

Usage:
    python scripts/evaluate_mmlu.py --base-only     # Evaluate base model only
    python scripts/evaluate_mmlu.py --finetuned     # Evaluate fine-tuned model
    python scripts/evaluate_mmlu.py --both          # Compare both
"""

import argparse
import json
import os
import random
from datetime import datetime

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# MMLU subjects to sample from (balanced mix)
MMLU_SUBJECTS = [
    "high_school_computer_science",
    "college_computer_science",
    "computer_security",
    "machine_learning",
    "abstract_algebra",
    "high_school_mathematics",
]


def load_mmlu_subset(num_questions: int = 100) -> list[dict]:
    """Load a balanced subset of MMLU questions."""
    print(f"Loading MMLU subset ({num_questions} questions)...")
    
    questions = []
    questions_per_subject = num_questions // len(MMLU_SUBJECTS)
    
    for subject in MMLU_SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
            # Sample questions from this subject
            indices = random.sample(range(len(ds)), min(questions_per_subject, len(ds)))
            for idx in indices:
                item = ds[idx]
                questions.append({
                    "subject": subject,
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],  # 0, 1, 2, or 3
                })
        except Exception as e:
            print(f"Warning: Could not load {subject}: {e}")
    
    random.shuffle(questions)
    print(f"Loaded {len(questions)} questions from {len(MMLU_SUBJECTS)} subjects")
    return questions[:num_questions]


def format_mmlu_prompt(question: dict) -> str:
    """Format an MMLU question as a prompt."""
    choices = question["choices"]
    choice_letters = ["A", "B", "C", "D"]
    
    prompt = f"Question: {question['question']}\n\n"
    for i, (letter, choice) in enumerate(zip(choice_letters, choices)):
        prompt += f"{letter}. {choice}\n"
    prompt += "\nAnswer with just the letter (A, B, C, or D):"
    
    return prompt


def extract_answer(response: str) -> str:
    """Extract the answer letter from model response."""
    response = response.strip().upper()
    
    # Direct letter
    if response in ["A", "B", "C", "D"]:
        return response
    
    # First character
    if response and response[0] in ["A", "B", "C", "D"]:
        return response[0]
    
    # Search for letter pattern
    for char in response:
        if char in ["A", "B", "C", "D"]:
            return char
    
    return "X"  # Invalid


def evaluate_model_on_mmlu(
    model,
    tokenizer,
    questions: list[dict],
    model_name: str,
) -> dict:
    """Evaluate a model on MMLU questions."""
    
    correct = 0
    results = []
    
    for q in tqdm(questions, desc=f"Evaluating {model_name}"):
        prompt = format_mmlu_prompt(q)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predicted = extract_answer(generated)
        expected = ["A", "B", "C", "D"][q["answer"]]
        is_correct = predicted == expected
        
        if is_correct:
            correct += 1
        
        results.append({
            "subject": q["subject"],
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
        })
    
    accuracy = correct / len(questions) if questions else 0
    
    # Per-subject breakdown
    subject_scores = {}
    for subject in MMLU_SUBJECTS:
        subject_results = [r for r in results if r["subject"] == subject]
        if subject_results:
            subject_scores[subject] = sum(r["correct"] for r in subject_results) / len(subject_results)
    
    return {
        "model": model_name,
        "total": len(questions),
        "correct": correct,
        "accuracy": accuracy,
        "subject_scores": subject_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="MMLU evaluation for catastrophic forgetting check")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--adapter", default="mwill-AImission/SecureCLI-Tuner-V2")
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--base-only", action="store_true", help="Evaluate base model only")
    parser.add_argument("--finetuned", action="store_true", help="Evaluate fine-tuned model only")
    parser.add_argument("--both", action="store_true", help="Evaluate both models")
    parser.add_argument("--output-dir", default="evaluation/results")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    if not any([args.base_only, args.finetuned, args.both]):
        args.both = True  # Default to both
    
    # Load questions
    questions = load_mmlu_subset(args.num_questions)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(questions),
        "seed": args.seed,
    }
    
    # Load tokenizer (shared)
    print(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Evaluate base model
    if args.base_only or args.both:
        print(f"\nLoading base model: {args.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        base_results = evaluate_model_on_mmlu(base_model, tokenizer, questions, "base")
        results["base_model"] = base_results
        
        print(f"\nBase Model MMLU Accuracy: {base_results['accuracy']:.1%}")
        
        # Free memory
        del base_model
        torch.cuda.empty_cache()
    
    # Evaluate fine-tuned model
    if args.finetuned or args.both:
        print(f"\nLoading fine-tuned model: {args.adapter}")
        ft_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        ft_model = PeftModel.from_pretrained(ft_model, args.adapter)
        
        ft_results = evaluate_model_on_mmlu(ft_model, tokenizer, questions, "fine-tuned")
        results["finetuned_model"] = ft_results
        
        print(f"\nFine-tuned Model MMLU Accuracy: {ft_results['accuracy']:.1%}")
    
    # Compare if both evaluated
    if args.both and "base_model" in results and "finetuned_model" in results:
        base_acc = results["base_model"]["accuracy"]
        ft_acc = results["finetuned_model"]["accuracy"]
        delta = ft_acc - base_acc
        
        print("\n" + "=" * 60)
        print("CATASTROPHIC FORGETTING CHECK")
        print("=" * 60)
        print(f"Base Model:      {base_acc:.1%}")
        print(f"Fine-tuned:      {ft_acc:.1%}")
        print(f"Delta:           {delta:+.1%}")
        
        if delta >= -0.05:
            print("\n✅ PASSED: No significant capability loss detected")
            results["catastrophic_forgetting_check"] = "PASSED"
        else:
            print(f"\n⚠️ WARNING: Accuracy dropped by {abs(delta):.1%}")
            results["catastrophic_forgetting_check"] = "WARNING"
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "mmlu_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
