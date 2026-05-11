from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from core.context import detect_context_flags
from core.rules import run_rule_checks
from core.scoring import ARTIFACT_RULES, STRONG_RULES
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


def category_metrics(
    rows: list[dict[str, object]],
    preds: list[int],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[int]]] = {}

    for row, pred in zip(rows, preds):
        category = str(row["category"])
        label = int(row["label"])

        if category not in grouped:
            grouped[category] = {
                "labels": [],
                "preds": [],
            }

        grouped[category]["labels"].append(label)
        grouped[category]["preds"].append(pred)

    results: dict[str, dict[str, float]] = {}

    for category, values in grouped.items():
        labels = values["labels"]
        category_preds = values["preds"]

        correct = sum(
            1 for label, pred in zip(labels, category_preds) if label == pred
        )

        results[category] = {
            "count": float(len(labels)),
            "accuracy": float(correct / len(labels)) if labels else 0.0,
            "positive_rate": float(sum(category_preds) / len(category_preds))
            if category_preds
            else 0.0,
            "true_label": float(labels[0]) if labels else -1.0,
        }

    return results


def train_supervised_model(
    model_type: str,
    train_rows: list[dict[str, object]],
    tokenizer: TokenizerWrapper,
) -> PromptSupervisedModel:
    train_prompts = [str(row["prompt"]) for row in train_rows]
    train_labels = [int(row["label"]) for row in train_rows]

    train_features = [
        extract_supervised_features(prompt, tokenizer) for prompt in train_prompts
    ]

    model = PromptSupervisedModel(model_type=model_type)
    model.fit(train_features, train_labels)

    return model


def predict_supervised(
    model: PromptSupervisedModel,
    prompts: list[str],
    tokenizer: TokenizerWrapper,
    threshold: float,
) -> tuple[list[int], list[float]]:
    features = [
        extract_supervised_features(prompt, tokenizer) for prompt in prompts
    ]

    probabilities = model.predict_proba(features)
    preds = [1 if prob >= threshold else 0 for prob in probabilities]

    return preds, probabilities


def combined_rules_plus_supervised_prediction(
    prompt: str,
    supervised_probability: float,
    probability_threshold: float,
) -> dict[str, object]:
    rules = run_rule_checks(prompt)
    rule_hits = set(rules["rule_hits"])
    context = detect_context_flags(prompt)

    strong_hits = sorted(hit for hit in rule_hits if hit in STRONG_RULES)
    artifact_hits = sorted(hit for hit in rule_hits if hit in ARTIFACT_RULES)

    has_strong_rule = len(strong_hits) > 0
    has_artifact_rule = len(artifact_hits) > 0
    model_flags = supervised_probability >= probability_threshold

    benign_technical_strength = int(context["benign_technical_strength"])
    analysis_context = bool(context["analysis_context"])

    # High-severity rules remain decisive unless the prompt clearly frames the text
    # as analysis or quoted content. This keeps explicit attacks high risk while
    # reducing obvious false positives from security discussion prompts.
    if has_strong_rule and not analysis_context:
        suspicious = True
    elif has_strong_rule and supervised_probability >= 0.70:
        suspicious = True
    elif has_artifact_rule and supervised_probability >= probability_threshold:
        suspicious = True
    elif model_flags:
        if benign_technical_strength >= 2 and supervised_probability < 0.80:
            suspicious = False
        else:
            suspicious = True
    else:
        suspicious = False

    if suspicious and (has_strong_rule or supervised_probability >= 0.85):
        risk_band = "high"
    elif suspicious:
        risk_band = "medium"
    else:
        risk_band = "low"

    return {
        "prediction": 1 if suspicious else 0,
        "risk_band": risk_band,
        "supervised_probability": supervised_probability,
        "rule_hits": sorted(rule_hits),
        "strong_rule_hits": strong_hits,
        "artifact_rule_hits": artifact_hits,
        "context_flags": context,
    }


def evaluate_thresholds(
    test_rows: list[dict[str, object]],
    probabilities: list[float],
    thresholds: list[float],
) -> dict[str, object]:
    labels = [int(row["label"]) for row in test_rows]
    prompts = [str(row["prompt"]) for row in test_rows]

    results: dict[str, object] = {}

    for threshold in thresholds:
        supervised_preds = [1 if prob >= threshold else 0 for prob in probabilities]
        combined_outputs = [
            combined_rules_plus_supervised_prediction(
                prompt=prompt,
                supervised_probability=prob,
                probability_threshold=threshold,
            )
            for prompt, prob in zip(prompts, probabilities)
        ]
        combined_preds = [int(output["prediction"]) for output in combined_outputs]

        threshold_key = f"{threshold:.2f}"

        results[threshold_key] = {
            "supervised_only": summarize_metrics(labels, supervised_preds),
            "combined_rules_plus_supervised": summarize_metrics(
                labels,
                combined_preds,
            ),
            "category_metrics_supervised_only": category_metrics(
                test_rows,
                supervised_preds,
            ),
            "category_metrics_combined": category_metrics(
                test_rows,
                combined_preds,
            ),
        }

    return results


def write_per_prompt_outputs(
    output_path: str,
    test_rows: list[dict[str, object]],
    probabilities: list[float],
    combined_outputs: list[dict[str, object]],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "prompt",
        "label",
        "category",
        "supervised_probability",
        "combined_prediction",
        "risk_band",
        "rule_hits",
        "context_flags",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row, probability, combined in zip(
            test_rows,
            probabilities,
            combined_outputs,
        ):
            writer.writerow(
                {
                    "prompt": row["prompt"],
                    "label": row["label"],
                    "category": row["category"],
                    "supervised_probability": probability,
                    "combined_prediction": combined["prediction"],
                    "risk_band": combined["risk_band"],
                    "rule_hits": json.dumps(combined["rule_hits"]),
                    "context_flags": json.dumps(combined["context_flags"]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run advanced PromptShield model analysis."
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/prompts_labeled.csv",
    )
    parser.add_argument(
        "--model-type",
        default="hist_gradient_boosting",
        choices=["logistic_regression", "hist_gradient_boosting"],
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

    labels = [int(row["label"]) for row in rows]

    train_rows, test_rows = train_test_split(
        rows,
        test_size=args.test_size,
        random_state=42,
        stratify=labels,
    )

    tokenizer = TokenizerWrapper()
    model = train_supervised_model(
        model_type=args.model_type,
        train_rows=train_rows,
        tokenizer=tokenizer,
    )

    test_prompts = [str(row["prompt"]) for row in test_rows]
    test_labels = [int(row["label"]) for row in test_rows]

    _, probabilities = predict_supervised(
        model=model,
        prompts=test_prompts,
        tokenizer=tokenizer,
        threshold=0.50,
    )

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    threshold_results = evaluate_thresholds(
        test_rows=test_rows,
        probabilities=probabilities,
        thresholds=thresholds,
    )

    best_threshold = None
    best_f1 = -1.0

    for threshold_key, result in threshold_results.items():
        combined_metrics = result["combined_rules_plus_supervised"]
        attack_f1 = float(combined_metrics["attack_f1"])

        if attack_f1 > best_f1:
            best_f1 = attack_f1
            best_threshold = threshold_key

    selected_threshold = float(best_threshold) if best_threshold is not None else 0.50

    combined_outputs = [
        combined_rules_plus_supervised_prediction(
            prompt=prompt,
            supervised_probability=prob,
            probability_threshold=selected_threshold,
        )
        for prompt, prob in zip(test_prompts, probabilities)
    ]
    combined_preds = [int(output["prediction"]) for output in combined_outputs]

    print("\n=== PromptShield Advanced Analysis ===")
    print(f"Dataset rows: {len(rows)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Model type: {args.model_type}")

    print("\n=== Threshold Sweep ===")
    for threshold_key, result in threshold_results.items():
        supervised_metrics = result["supervised_only"]
        combined_metrics = result["combined_rules_plus_supervised"]

        print(f"\nThreshold: {threshold_key}")
        print(
            "Supervised only: "
            f"accuracy={supervised_metrics['accuracy']:.3f}, "
            f"precision={supervised_metrics['attack_precision']:.3f}, "
            f"recall={supervised_metrics['attack_recall']:.3f}, "
            f"f1={supervised_metrics['attack_f1']:.3f}"
        )
        print(
            "Combined rules + supervised: "
            f"accuracy={combined_metrics['accuracy']:.3f}, "
            f"precision={combined_metrics['attack_precision']:.3f}, "
            f"recall={combined_metrics['attack_recall']:.3f}, "
            f"f1={combined_metrics['attack_f1']:.3f}"
        )

    print("\n=== Selected Combined System ===")
    print(f"Selected threshold: {selected_threshold:.2f}")
    print(
        classification_report(
            test_labels,
            combined_preds,
            digits=3,
            zero_division=0,
        )
    )

    print("\n=== Category-Level Combined Accuracy ===")
    selected_key = f"{selected_threshold:.2f}"
    category_results = threshold_results[selected_key]["category_metrics_combined"]

    for category, metrics in sorted(category_results.items()):
        print(
            f"{category}: "
            f"count={int(metrics['count'])}, "
            f"accuracy={metrics['accuracy']:.3f}, "
            f"positive_rate={metrics['positive_rate']:.3f}"
        )

    result_path = Path(args.results_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    advanced_output = {
        "dataset": args.dataset,
        "model_type": args.model_type,
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "selected_threshold": selected_threshold,
        "threshold_results": threshold_results,
    }

    json_output = result_path / "advanced_analysis.json"
    csv_output = result_path / "advanced_per_prompt_outputs.csv"

    json_output.write_text(json.dumps(advanced_output, indent=2), encoding="utf-8")

    write_per_prompt_outputs(
        output_path=str(csv_output),
        test_rows=test_rows,
        probabilities=probabilities,
        combined_outputs=combined_outputs,
    )

    print(f"\nWrote advanced analysis JSON to: {json_output}")
    print(f"Wrote per-prompt outputs CSV to: {csv_output}")


if __name__ == "__main__":
    main()