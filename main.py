from __future__ import annotations

import argparse

from core.pipeline import (
    load_pipeline_artifacts,
    load_prompts,
    train_isolation_forest_pipeline,
)


def print_result(result: dict[str, object]) -> None:
    scoring = result["scoring"]
    rules = result["rules"]

    print("\n=== PromptShield Result ===")
    print(f"Prompt: {result['prompt']}")
    print(f"Suspicious: {scoring['suspicious']}")
    print(f"Risk band: {scoring['risk_band']}")
    print(f"Anomaly score: {scoring['anomaly_score']:.6f}")
    print(f"Anomaly band: {scoring['anomaly_band']}")
    print(f"Rule hits: {rules['rule_hits']}")
    print(f"Context flags: {scoring['context_flags']}")
    print(f"Thresholds: {result['thresholds']}")
    print(f"Features: {result['features']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PromptShield CLI")
    parser.add_argument("--prompt", help="Prompt string to score")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--benign-file", default="data/prompts/benign.txt")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain from benign prompts instead of loading saved artifacts.",
    )
    args = parser.parse_args()

    if not args.prompt:
        raise ValueError("Use --prompt to provide a prompt for scoring.")

    if args.retrain:
        pipeline = train_isolation_forest_pipeline(benign_file=args.benign_file)
    else:
        pipeline = load_pipeline_artifacts(artifact_dir=args.artifact_dir)

    result = pipeline.score_prompt(args.prompt)
    print_result(result)


if __name__ == "__main__":
    main()
