from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from core.features import extract_features
from core.model import PromptAnomalyModel
from core.rules import run_rule_checks
from core.scoring import combine_evidence
from core.supervised import PromptSupervisedModel, extract_supervised_features
from core.tokenizer import TokenizerWrapper


def load_labeled_dataset(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(
                {
                    "prompt": row["prompt"],
                    "label": int(row["label"]),
                    "source_split": row.get("source_split", ""),
                    "category": row.get("category", ""),
                    "source_file": row.get("source_file", ""),
                }
            )

    return rows


def summarize_metrics(labels: list[int], preds: list[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "attack_precision": float(precision),
        "attack_recall": float(recall),
        "attack_f1": float(f1),
    }


def build_base_features(
    prompts: list[str],
    tokenizer: TokenizerWrapper,
) -> list[dict[str, float]]:
    features = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        features.append(extract_features(prompt, tokens))

    return features


def train_isolation_forest_from_train_split(
    train_rows: list[dict[str, object]],
    tokenizer: TokenizerWrapper,
) -> tuple[PromptAnomalyModel, dict[str, float]]:
    benign_prompts = [
        str(row["prompt"]) for row in train_rows if int(row["label"]) == 0
    ]

    if not benign_prompts:
        raise ValueError("No benign prompts available for Isolation Forest training.")

    benign_features = build_base_features(benign_prompts, tokenizer)

    model = PromptAnomalyModel()
    model.fit(benign_features)
    thresholds = model.calibrate_thresholds(benign_features)

    return model, thresholds


def evaluate_rule_only(test_prompts: list[str]) -> list[int]:
    preds = []

    for prompt in test_prompts:
        rules = run_rule_checks(prompt)
        preds.append(1 if int(rules["rule_count"]) > 0 else 0)

    return preds


def evaluate_isolation_only(
    test_prompts: list[str],
    tokenizer: TokenizerWrapper,
    model: PromptAnomalyModel,
    thresholds: dict[str, float],
) -> list[int]:
    preds = []

    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt)
        features = extract_features(prompt, tokens)
        anomaly_score = model.anomaly_score(features)
        preds.append(1 if anomaly_score >= thresholds["p90"] else 0)

    return preds


def evaluate_combined_isolation(
    test_prompts: list[str],
    tokenizer: TokenizerWrapper,
    model: PromptAnomalyModel,
    thresholds: dict[str, float],
) -> list[int]:
    preds = []

    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt)
        features = extract_features(prompt, tokens)
        anomaly_score = model.anomaly_score(features)
        rules = run_rule_checks(prompt)

        combined = combine_evidence(
            text=prompt,
            anomaly_score=anomaly_score,
            rule_hits=rules["rule_hits"],
            thresholds=thresholds,
        )

        preds.append(1 if bool(combined["suspicious"]) else 0)

    return preds


def evaluate_supervised_model(
    model_type: str,
    train_rows: list[dict[str, object]],
    test_prompts: list[str],
    tokenizer: TokenizerWrapper,
) -> list[int]:
    train_prompts = [str(row["prompt"]) for row in train_rows]
    train_labels = [int(row["label"]) for row in train_rows]

    train_features = [
        extract_supervised_features(prompt, tokenizer) for prompt in train_prompts
    ]
    test_features = [
        extract_supervised_features(prompt, tokenizer) for prompt in test_prompts
    ]

    model = PromptSupervisedModel(model_type=model_type)
    model.fit(train_features, train_labels)

    return model.predict(test_features)


def print_and_store_result(
    name: str,
    labels: list[int],
    preds: list[int],
    results: dict[str, object],
) -> None:
    metrics = summarize_metrics(labels, preds)
    report = classification_report(labels, preds, digits=3, zero_division=0)

    results[name] = {
        **metrics,
        "classification_report": report,
    }

    print(f"\n=== {name} ===")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PromptShield rule, anomaly, supervised, and combined models."
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/prompts_labeled.csv",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
    )
    args = parser.parse_args()

    rows = load_labeled_dataset(args.dataset)

    if len(rows) < 20:
        raise ValueError(
            "Dataset is too small. Run scripts/build_dataset.py first."
        )

    labels = [int(row["label"]) for row in rows]

    train_rows, test_rows = train_test_split(
        rows,
        test_size=args.test_size,
        random_state=42,
        stratify=labels,
    )

    test_prompts = [str(row["prompt"]) for row in test_rows]
    test_labels = [int(row["label"]) for row in test_rows]

    tokenizer = TokenizerWrapper()
    if_model, thresholds = train_isolation_forest_from_train_split(
        train_rows=train_rows,
        tokenizer=tokenizer,
    )

    results: dict[str, object] = {
        "dataset": args.dataset,
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "thresholds": thresholds,
    }

    print("\n=== PromptShield Model Comparison ===")
    print(f"Dataset rows: {len(rows)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Test rows: {len(test_rows)}")

    rule_preds = evaluate_rule_only(test_prompts)
    print_and_store_result("rule_only", test_labels, rule_preds, results)

    if_preds = evaluate_isolation_only(
        test_prompts=test_prompts,
        tokenizer=tokenizer,
        model=if_model,
        thresholds=thresholds,
    )
    print_and_store_result("isolation_forest_p90", test_labels, if_preds, results)

    combined_if_preds = evaluate_combined_isolation(
        test_prompts=test_prompts,
        tokenizer=tokenizer,
        model=if_model,
        thresholds=thresholds,
    )
    print_and_store_result(
        "combined_rules_plus_isolation",
        test_labels,
        combined_if_preds,
        results,
    )

    logistic_preds = evaluate_supervised_model(
        model_type="logistic_regression",
        train_rows=train_rows,
        test_prompts=test_prompts,
        tokenizer=tokenizer,
    )
    print_and_store_result(
        "supervised_logistic_regression",
        test_labels,
        logistic_preds,
        results,
    )

    hgb_preds = evaluate_supervised_model(
        model_type="hist_gradient_boosting",
        train_rows=train_rows,
        test_prompts=test_prompts,
        tokenizer=tokenizer,
    )
    print_and_store_result(
        "supervised_hist_gradient_boosting",
        test_labels,
        hgb_preds,
        results,
    )

    result_path = Path(args.results_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    output_file = result_path / "model_comparison_metrics.json"
    output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nWrote model comparison results to: {output_file}")


if __name__ == "__main__":
    main()