import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table

def load_metrics(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Compare V1 vs V2 results")
    parser.add_argument("--v2-results", default="evaluation/semantic/metrics.json", help="Path to V2 metrics.json")
    args = parser.parse_args()

    console = Console()
    v2_metrics = load_metrics(Path(args.v2_results))
    
    # Baseline V1 stats (from technical report)
    v1_metrics = {
        "exact_match_rate": 0.1322,
        "functional_match_rate": 0.1322, # V1 had no functional eval
        "command_only_rate": 0.994,
        "security_blocking": 0.57  # Adversarial safe rate
    }

    table = Table(title="SecureCLI-Tuner Comparison: V1 vs V2", header_style="bold magenta")
    
    table.add_column("Metric", style="cyan")
    table.add_column("V1 (Baseline)", justify="right")
    table.add_column("V2 (Measured)", justify="right")
    table.add_column("Delta", style="green", justify="right")
    table.add_column("Status", justify="center")

    if not v2_metrics:
        console.print("[bold yellow]WARNING:[/bold yellow] V2 metrics not found at " + args.v2_results)
        console.print("Run 'python scripts/evaluate_semantic.py --checkpoint ...' first.\n")
        # Show targets instead
        results = [
            ["Exact Match Rate", f"{v1_metrics['exact_match_rate']:.1%}", "PENDING", "-", "⏳"],
            ["Functional Match", f"{v1_metrics['functional_match_rate']:.1%}", "PENDING", "-", "⏳"],
            ["Security Blocking", f"{v1_metrics['security_blocking']:.1%}", "PENDING", "-", "⏳"]
        ]
    else:
        def fmt(val): return f"{val:.1%}"
        def diff(v2, v1): 
            d = v2 - v1
            return f"+{d:.1%}" if d > 0 else f"{d:.1%}"

        results = [
            ["Exact Match Rate", fmt(v1_metrics["exact_match_rate"]), fmt(v2_metrics.get("exact_match_rate", 0)), 
             diff(v2_metrics.get("exact_match_rate", 0), v1_metrics["exact_match_rate"]), "✅" if v2_metrics.get("exact_match_rate", 0) >= v1_metrics["exact_match_rate"] else "❌"],
            ["Functional Match", fmt(v1_metrics["functional_match_rate"]), fmt(v2_metrics.get("functional_match_rate", 0)),
             diff(v2_metrics.get("functional_match_rate", 0), v1_metrics["functional_match_rate"]), "✅"],
            ["Security Blocking", fmt(v1_metrics["security_blocking"]), "PENDING*", "-", "🛡️"]
        ]

    for row in results:
        table.add_row(*row)

    console.print(table)
    if not v2_metrics:
        console.print("\n[bold red]Rigor Check:[/bold red] Use actual measured metrics to populate this table.")
    else:
        console.print("\n[bold green]Rigor Check:[/bold green] Metrics loaded from " + args.v2_results)
        console.print("*Security blocking must be verified via 'pytest tests/eval/test_adversarial.py'")

if __name__ == "__main__":
    main()
