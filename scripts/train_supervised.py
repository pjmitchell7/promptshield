from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from core.supervised import (
    PromptSupervisedModel,
    extract_supervised_features,
    supervised_feature_names,
)
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


def train_and_evaluate(
    model_type: str,
    train_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    tokenizer: TokenizerWrapper,
) -> tuple[PromptSupervisedModel, dict[str, object]]:
    train_prompts = [str(row["prompt"]) for row in train_rows]
    train_labels = [int(row["label"]) for row in train_rows]

    test_prompts = [str(row["prompt"]) for row in test_rows]
    test_labels = [int(row["label"]) for row in test_rows]

    train_features = [
        extract_supervised_features(prompt, tokenizer) for prompt in train_prompts
    ]
    test_features = [
        extract_supervised_features(prompt, tokenizer) for prompt in test_prompts
    ]

    model = PromptSupervisedModel(model_type=model_type)
    model.fit(train_features, train_labels)

    preds = model.predict(test_features)

    metrics = summarize_metrics(test_labels, preds)
    metrics["model_type"] = model_type
    metrics["train_size"] = len(train_rows)
    metrics["test_size"] = len(test_rows)
    metrics["classification_report"] = classification_report(
        test_labels,
        preds,
        digits=3,
        zero_division=0,
    )
    metrics["confusion_matrix"] = confusion_matrix(test_labels, preds).tolist()

    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train supervised PromptShield classifiers."
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/prompts_labeled.csv",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts",
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
            "Dataset is too small for supervised training. Run scripts/build_dataset.py first."
        )

    labels = [int(row["label"]) for row in rows]

    train_rows, test_rows = train_test_split(
        rows,
        test_size=args.test_size,
        random_state=42,
        stratify=labels,
    )

    tokenizer = TokenizerWrapper()

    artifact_path = Path(args.artifact_dir)
    result_path = Path(args.results_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)

    model_types = [
        "logistic_regression",
        "hist_gradient_boosting",
    ]

    all_metrics: list[dict[str, object]] = []
    trained_models: dict[str, PromptSupervisedModel] = {}

    print("\n=== Supervised PromptShield Training ===")
    print(f"Dataset: {args.dataset}")
    print(f"Rows: {len(rows)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Test rows: {len(test_rows)}")

    for model_type in model_types:
        model, metrics = train_and_evaluate(
            model_type=model_type,
            train_rows=train_rows,
            test_rows=test_rows,
            tokenizer=tokenizer,
        )

        trained_models[model_type] = model
        all_metrics.append(metrics)

        joblib.dump(model, artifact_path / f"supervised_{model_type}.joblib")

        print(f"\n=== {model_type} ===")
        print(metrics["classification_report"])
        print("Confusion matrix:")
        print(metrics["confusion_matrix"])

    best_metrics = max(all_metrics, key=lambda item: float(item["attack_f1"]))
    best_model_type = str(best_metrics["model_type"])
    best_model = trained_models[best_model_type]

    joblib.dump(best_model, artifact_path / "supervised_best.joblib")

    metadata = {
        "best_model_type": best_model_type,
        "dataset": args.dataset,
        "feature_names": supervised_feature_names(),
        "all_metrics": all_metrics,
    }

    (artifact_path / "supervised_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    (result_path / "supervised_metrics.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n=== Best Supervised Model ===")
    print(f"Best model type: {best_model_type}")
    print(
        "Attack F1: "
        f"{float(best_metrics['attack_f1']):.3f}, "
        "Accuracy: "
        f"{float(best_metrics['accuracy']):.3f}"
    )


if __name__ == "__main__":
    main()