from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from core.features import extract_features
from core.model import PromptAnomalyModel
from core.rules import run_rule_checks
from core.scoring import combine_evidence
from core.tokenizer import TokenizerWrapper


def load_prompts(path: str) -> list[str]:
    raw_text = Path(path).read_text(encoding="utf-8")

    if "===PROMPT===" in raw_text:
        chunks = [chunk.strip() for chunk in raw_text.split("===PROMPT===")]
        return [chunk for chunk in chunks if chunk]

    lines = [line.strip() for line in raw_text.splitlines()]
    return [line for line in lines if line]


def build_feature(prompt: str, tokenizer: TokenizerWrapper) -> dict[str, float]:
    tokens = tokenizer.encode(prompt)
    return extract_features(prompt, tokens)


def train_on_benign(tokenizer: TokenizerWrapper) -> tuple[PromptAnomalyModel, list[dict[str, float]], list[str]]:
    benign_prompts = load_prompts("data/prompts/benign.txt")
    benign_features = [build_feature(prompt, tokenizer) for prompt in benign_prompts]

    model = PromptAnomalyModel()
    model.fit(benign_features)
    return model, benign_features, benign_prompts


def shorten_prompt(prompt: str, max_len: int = 90) -> str:
    flat = " ".join(prompt.split())
    if len(flat) <= max_len:
        return flat
    return flat[: max_len - 3] + "..."


def percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(np.array(values, dtype=float), pct))


def print_score_summary(name: str, scores: list[float]) -> None:
    print(f"\n=== {name} anomaly score summary ===")
    print(f"count: {len(scores)}")
    print(f"min:   {min(scores):.6f}")
    print(f"max:   {max(scores):.6f}")
    print(f"mean:  {float(np.mean(scores)):.6f}")
    print(f"p50:   {percentile(scores, 50):.6f}")
    print(f"p90:   {percentile(scores, 90):.6f}")
    print(f"p95:   {percentile(scores, 95):.6f}")
    print(f"p99:   {percentile(scores, 99):.6f}")


def evaluate() -> None:
    tokenizer = TokenizerWrapper()
    model, benign_features, benign_prompts = train_on_benign(tokenizer)
    thresholds = model.calibrate_thresholds(benign_features)

    benign = benign_prompts
    adversarial = load_prompts("data/prompts/adversarial.txt")
    perturbed = load_prompts("data/prompts/perturbed.txt")

    all_examples: list[dict[str, object]] = []

    for prompt in benign:
        all_examples.append({"prompt": prompt, "label": 0, "split": "benign"})

    for prompt in adversarial:
        all_examples.append({"prompt": prompt, "label": 1, "split": "adversarial"})

    for prompt in perturbed:
        all_examples.append({"prompt": prompt, "label": 1, "split": "perturbed"})

    benign_scores: list[float] = []
    adversarial_scores: list[float] = []
    perturbed_scores: list[float] = []

    for example in all_examples:
        prompt = str(example["prompt"])
        features = build_feature(prompt, tokenizer)
        anomaly_score = model.anomaly_score(features)

        example["features"] = features
        example["anomaly_score"] = anomaly_score

        split = str(example["split"])
        if split == "benign":
            benign_scores.append(anomaly_score)
        elif split == "adversarial":
            adversarial_scores.append(anomaly_score)
        else:
            perturbed_scores.append(anomaly_score)

    print_score_summary("Benign", benign_scores)
    print_score_summary("Adversarial", adversarial_scores)
    print_score_summary("Perturbed", perturbed_scores)

    print("\n=== Calibrated anomaly thresholds from benign scores ===")
    print(f"p90 threshold: {thresholds['p90']:.6f}")
    print(f"p95 threshold: {thresholds['p95']:.6f}")
    print(f"p99 threshold: {thresholds['p99']:.6f}")

    labels = [int(example["label"]) for example in all_examples]

    rule_only_preds = []
    anomaly_only_preds_p90 = []
    anomaly_only_preds_p95 = []
    combined_preds = []

    print("\n=== Per-prompt inspection ===")
    print(
        "split".ljust(12),
        "label".ljust(6),
        "anom".ljust(10),
        "rules".ljust(6),
        "a90".ljust(5),
        "a95".ljust(5),
        "combo".ljust(6),
        "preview",
    )
    print("-" * 120)

    for example in all_examples:
        prompt = str(example["prompt"])
        label = int(example["label"])
        split = str(example["split"])
        anomaly_score = float(example["anomaly_score"])
        features = dict(example["features"])

        rules = run_rule_checks(prompt)
        combined = combine_evidence(
            text=prompt,
            anomaly_score=anomaly_score,
            rule_hits=rules["rule_hits"],
            thresholds=thresholds,
        )

        rule_pred = 1 if rules["rule_count"] > 0 else 0
        anomaly_pred_p90 = 1 if anomaly_score >= thresholds["p90"] else 0
        anomaly_pred_p95 = 1 if anomaly_score >= thresholds["p95"] else 0
        combined_pred = 1 if combined["suspicious"] else 0

        rule_only_preds.append(rule_pred)
        anomaly_only_preds_p90.append(anomaly_pred_p90)
        anomaly_only_preds_p95.append(anomaly_pred_p95)
        combined_preds.append(combined_pred)

        print(
            split.ljust(12),
            str(label).ljust(6),
            f"{anomaly_score:.6f}".ljust(10),
            str(rules["rule_count"]).ljust(6),
            str(anomaly_pred_p90).ljust(5),
            str(anomaly_pred_p95).ljust(5),
            str(combined_pred).ljust(6),
            shorten_prompt(prompt),
        )

        if rules["rule_hits"]:
            print(" " * 49 + f"rule_hits={rules['rule_hits']}")

        if combined["context_flags"]["benign_technical_context"] or combined["context_flags"]["analysis_context"]:
            print(" " * 49 + f"context_flags={combined['context_flags']}")

        if (
            features["zero_width_count"] > 0
            or features["homoglyph_count"] > 0
            or features["non_ascii_ratio"] > 0
        ):
            print(
                " " * 49
                + "feature_flags="
                + str(
                    {
                        "zero_width_count": features["zero_width_count"],
                        "homoglyph_count": features["homoglyph_count"],
                        "non_ascii_ratio": round(features["non_ascii_ratio"], 6),
                    }
                )
            )

    print("\n=== Rule-only baseline ===")
    print(classification_report(labels, rule_only_preds, digits=3, zero_division=0))

    print("\n=== Anomaly-only baseline (p90 threshold) ===")
    print(classification_report(labels, anomaly_only_preds_p90, digits=3, zero_division=0))

    print("\n=== Anomaly-only baseline (p95 threshold) ===")
    print(classification_report(labels, anomaly_only_preds_p95, digits=3, zero_division=0))

    print("\n=== Combined PromptShield baseline ===")
    print(classification_report(labels, combined_preds, digits=3, zero_division=0))


if __name__ == "__main__":
    evaluate()