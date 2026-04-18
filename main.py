from __future__ import annotations

import argparse
from pathlib import Path

from core.features import extract_features
from core.model import PromptAnomalyModel
from core.rules import run_rule_checks
from core.scoring import combine_evidence
from core.tokenizer import TokenizerWrapper


def load_prompts(file_path: str) -> list[str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    raw_text = path.read_text(encoding="utf-8")

    if "===PROMPT===" in raw_text:
        chunks = [chunk.strip() for chunk in raw_text.split("===PROMPT===")]
        return [chunk for chunk in chunks if chunk]

    lines = [line.strip() for line in raw_text.splitlines()]
    return [line for line in lines if line]


def build_feature_dicts(
    prompts: list[str],
    tokenizer: TokenizerWrapper,
) -> list[dict[str, float]]:
    feature_dicts = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        feature_dicts.append(extract_features(prompt, tokens))
    return feature_dicts


def train_model(benign_file: str) -> tuple[PromptAnomalyModel, dict[str, float]]:
    tokenizer = TokenizerWrapper()
    benign_prompts = load_prompts(benign_file)
    benign_features = build_feature_dicts(benign_prompts, tokenizer)

    model = PromptAnomalyModel()
    model.fit(benign_features)
    thresholds = model.calibrate_thresholds(benign_features)

    return model, thresholds


def score_prompt(
    prompt: str,
    model: PromptAnomalyModel,
    tokenizer: TokenizerWrapper,
    thresholds: dict[str, float],
) -> dict[str, object]:
    tokens = tokenizer.encode(prompt)
    feature_dict = extract_features(prompt, tokens)
    anomaly_score = model.anomaly_score(feature_dict)

    rules = run_rule_checks(prompt)
    evidence = combine_evidence(
        text=prompt,
        anomaly_score=anomaly_score,
        rule_hits=rules["rule_hits"],
        thresholds=thresholds,
    )

    return {
        "prompt": prompt,
        "tokens": tokens,
        "features": feature_dict,
        "rules": rules,
        "scoring": evidence,
        "thresholds": thresholds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PromptShield v1 CLI")
    parser.add_argument("--benign-file", default="data/prompts/benign.txt")
    parser.add_argument("--prompt", help="Prompt string to score")
    args = parser.parse_args()

    if not args.prompt:
        raise ValueError("Use --prompt to provide a prompt for scoring.")

    tokenizer = TokenizerWrapper()
    model, thresholds = train_model(args.benign_file)
    result = score_prompt(args.prompt, model, tokenizer, thresholds)

    print("\n=== PromptShield Result ===")
    print(f"Prompt: {result['prompt']}")
    print(f"Suspicious: {result['scoring']['suspicious']}")
    print(f"Risk band: {result['scoring']['risk_band']}")
    print(f"Anomaly score: {result['scoring']['anomaly_score']:.6f}")
    print(f"Anomaly band: {result['scoring']['anomaly_band']}")
    print(f"Rule hits: {result['rules']['rule_hits']}")
    print(f"Context flags: {result['scoring']['context_flags']}")
    print(f"Thresholds: {result['thresholds']}")
    print(f"Features: {result['features']}")


if __name__ == "__main__":
    main()