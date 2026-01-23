#!/usr/bin/env python3
"""
Download and prepare all datasets for SecureCLI-Tuner training.

Datasets:
- prabhanshubhowal/natural_language_to_linux (Bash)
- Future: Git and Docker datasets

Run:
    python scripts/download_datasets.py
"""
import json
from pathlib import Path
from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"


def download_bash_dataset():
    """Download the primary Bash dataset."""
    print("=" * 60)
    print("Downloading: prabhanshubhowal/natural_language_to_linux")
    print("=" * 60)
    
    dataset = load_dataset(
        "prabhanshubhowal/natural_language_to_linux",
        trust_remote_code=False
    )
    
    # Get the train split
    if "train" in dataset:
        records = dataset["train"]
    else:
        records = dataset[next(iter(dataset.keys()))]
    
    print(f"  Downloaded: {len(records)} examples")
    
    # Save to raw directory
    output_path = RAW_DIR / "bash_nl_to_linux.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            # Normalize fields
            normalized = {
                "instruction": record.get("nl_command", ""),
                "output": record.get("bash_code", ""),
                "input": record.get("input", ""),
                "tool": "bash",
                "source": "prabhanshubhowal/natural_language_to_linux"
            }
            f.write(json.dumps(normalized, ensure_ascii=True) + "\n")
    
    print(f"  Saved to: {output_path}")
    return len(records)


def download_tldr_dataset():
    """Download TLDR pages for additional Bash examples."""
    print("=" * 60)
    print("Downloading: cheshire-cat-ai/tldr-pages")
    print("=" * 60)
    
    try:
        dataset = load_dataset(
            "cheshire-cat-ai/tldr-pages",
            trust_remote_code=False
        )
        
        if "train" in dataset:
            records = dataset["train"]
        else:
            records = dataset[next(iter(dataset.keys()))]
        
        print(f"  Downloaded: {len(records)} examples")
        
        # Filter for common/linux pages only
        output_path = RAW_DIR / "tldr_pages.jsonl"
        count = 0
        
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                # TLDR format: command name + description + example
                if "platform" in record and record["platform"] in ["common", "linux"]:
                    normalized = {
                        "instruction": record.get("description", ""),
                        "output": record.get("command", ""),
                        "input": "",
                        "tool": "bash",
                        "source": "cheshire-cat-ai/tldr-pages"
                    }
                    if normalized["instruction"] and normalized["output"]:
                        f.write(json.dumps(normalized, ensure_ascii=True) + "\n")
                        count += 1
        
        print(f"  Filtered to {count} bash examples")
        print(f"  Saved to: {output_path}")
        return count
    except Exception as e:
        print(f"  Warning: Could not download TLDR dataset: {e}")
        return 0


def main():
    """Download all datasets."""
    print("\n" + "=" * 60)
    print("SecureCLI-Tuner Dataset Download")
    print("=" * 60 + "\n")
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    total = 0
    
    # Primary Bash dataset
    total += download_bash_dataset()
    
    # Optional: TLDR pages
    total += download_tldr_dataset()
    
    print("\n" + "=" * 60)
    print(f"Download complete! Total examples: {total}")
    print(f"Raw data location: {RAW_DIR}")
    print("=" * 60)
    print("\nNext step: python scripts/prepare_training_data.py")


if __name__ == "__main__":
    main()
