from __future__ import annotations
from pathlib import Path

from sklearn.metrics import classification_report

from core.features import extract_features
from core.model import PromptAnomalyModel
from core.rules import run_rule_checks
from core.scoring import combine_scores
from core.tokenizer import TokenizerWrapper


def load_prompts(path: str) -> list[str]:
    raw_text = Path(path).read_text(encoding="utf-8")

    # If the file uses prompt separators, each chunk is one prompt.
    if "===PROMPT===" in raw_text:
        chunks = [chunk.strip() for chunk in raw_text.split("===PROMPT===")]
        return [chunk for chunk in chunks if chunk]

    # Fallback for older one-line prompt files.
    lines = [line.strip() for line in raw_text.splitlines()]
    return [line for line in lines if line]


def build_feature(prompt: str, tokenizer: TokenizerWrapper) -> dict[str, float]:
    tokens = tokenizer.encode(prompt)
    return extract_features(prompt, tokens)


def train_on_benign(tokenizer: TokenizerWrapper) -> PromptAnomalyModel:
    benign_prompts = load_prompts("data/prompts/benign.txt")
    benign_features = [build_feature(prompt, tokenizer) for prompt in benign_prompts]

    model = PromptAnomalyModel()
    model.fit(benign_features)
    return model


def evaluate() -> None:
    tokenizer = TokenizerWrapper()
    model = train_on_benign(tokenizer)

    benign = load_prompts("data/prompts/benign.txt")
    adversarial = load_prompts("data/prompts/adversarial.txt")
    perturbed = load_prompts("data/prompts/perturbed.txt")

    all_prompts = benign + adversarial + perturbed

    # 0 means benign, 1 means suspicious.
    labels = ([0] * len(benign)) + ([1] * len(adversarial)) + ([1] * len(perturbed))

    rule_only_preds = []
    anomaly_only_preds = []
    combined_preds = []

    for prompt in all_prompts:
        features = build_feature(prompt, tokenizer)
        anomaly_score = model.anomaly_score(features)
        rules = run_rule_checks(prompt)
        combined = combine_scores(
            anomaly_score=anomaly_score,
            rule_count=rules["rule_count"],
        )

        # These thresholds are starter baselines so I can compare system behavior quickly.
        rule_only_preds.append(1 if rules["rule_count"] > 0 else 0)
        anomaly_only_preds.append(1 if anomaly_score >= 0.35 else 0)
        combined_preds.append(1 if combined["risk_band"] in {"medium", "high"} else 0)

    print("\n=== Rule-only baseline ===")
    print(classification_report(labels, rule_only_preds, digits=3))

    print("\n=== Anomaly-only baseline ===")
    print(classification_report(labels, anomaly_only_preds, digits=3))

    print("\n=== Combined PromptShield ===")
    print(classification_report(labels, combined_preds, digits=3))


if __name__ == "__main__":
    evaluate()