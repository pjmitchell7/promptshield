from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import joblib

from core.features import extract_features
from core.model import PromptAnomalyModel
from core.pipeline import load_prompts
from core.rules import run_rule_checks
from core.scoring import combine_evidence
from core.supervised import (
    PromptSupervisedModel,
    extract_supervised_features,
)
from core.tokenizer import TokenizerWrapper


def load_labeled_prompts(dataset_path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    with Path(dataset_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(
                {
                    "prompt": row["prompt"],
                    "label": int(row["label"]),
                    "category": row.get("category", ""),
                }
            )

    return rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    return float(sorted_values[index])


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    total_ms = sum(latencies_ms)
    total_seconds = total_ms / 1000.0

    if total_seconds <= 0:
        throughput = 0.0
    else:
        throughput = len(latencies_ms) / total_seconds

    return {
        "count": float(len(latencies_ms)),
        "mean_ms": float(statistics.mean(latencies_ms)),
        "median_ms": float(statistics.median(latencies_ms)),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
        "throughput_prompts_per_second": float(throughput),
    }


def train_isolation_forest(
    prompts: list[str],
    labels: list[int],
    tokenizer: TokenizerWrapper,
) -> tuple[PromptAnomalyModel, dict[str, float]]:
    benign_prompts = [
        prompt for prompt, label in zip(prompts, labels) if label == 0
    ]

    benign_features = []

    for prompt in benign_prompts:
        tokens = tokenizer.encode(prompt)
        benign_features.append(extract_features(prompt, tokens))

    model = PromptAnomalyModel()
    model.fit(benign_features)
    thresholds = model.calibrate_thresholds(benign_features)

    return model, thresholds


def benchmark_rule_only(prompts: list[str]) -> list[float]:
    latencies = []

    for prompt in prompts:
        start = time.perf_counter()
        run_rule_checks(prompt)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    return latencies


def benchmark_isolation_only(
    prompts: list[str],
    tokenizer: TokenizerWrapper,
    model: PromptAnomalyModel,
) -> list[float]:
    latencies = []

    for prompt in prompts:
        start = time.perf_counter()
        tokens = tokenizer.encode(prompt)
        features = extract_features(prompt, tokens)
        model.anomaly_score(features)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    return latencies


def benchmark_combined_isolation(
    prompts: list[str],
    tokenizer: TokenizerWrapper,
    model: PromptAnomalyModel,
    thresholds: dict[str, float],
) -> list[float]:
    latencies = []

    for prompt in prompts:
        start = time.perf_counter()
        tokens = tokenizer.encode(prompt)
        features = extract_features(prompt, tokens)
        anomaly_score = model.anomaly_score(features)
        rules = run_rule_checks(prompt)

        combine_evidence(
            text=prompt,
            anomaly_score=anomaly_score,
            rule_hits=rules["rule_hits"],
            thresholds=thresholds,
        )

        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    return latencies


def benchmark_supervised(
    prompts: list[str],
    tokenizer: TokenizerWrapper,
    model: PromptSupervisedModel,
) -> list[float]:
    latencies = []

    for prompt in prompts:
        start = time.perf_counter()
        features = extract_supervised_features(prompt, tokenizer)
        model.predict([features])
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    return latencies


def create_prompt_length_sets(base_prompts: list[str]) -> dict[str, list[str]]:
    short_prompts = []
    medium_prompts = []
    long_prompts = []
    very_long_prompts = []

    for prompt in base_prompts:
        one_line = " ".join(prompt.split())

        short_prompts.append(one_line[:200])
        medium_prompts.append((one_line + " ") * 4)
        long_prompts.append((one_line + " ") * 16)
        very_long_prompts.append((one_line + " ") * 64)

    return {
        "short": short_prompts,
        "medium": medium_prompts,
        "long": long_prompts,
        "very_long": very_long_prompts,
    }


def write_csv_summary(
    results: dict[str, dict[str, dict[str, float]]],
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "variant",
        "prompt_length_group",
        "count",
        "mean_ms",
        "median_ms",
        "p95_ms",
        "p99_ms",
        "throughput_prompts_per_second",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for variant, length_results in results.items():
            for length_group, metrics in length_results.items():
                row = {
                    "variant": variant,
                    "prompt_length_group": length_group,
                    **metrics,
                }
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PromptShield runtime overhead."
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
        "--sample-size",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    rows = load_labeled_prompts(args.dataset)
    rows = rows[: args.sample_size]

    prompts = [str(row["prompt"]) for row in rows]
    labels = [int(row["label"]) for row in rows]

    tokenizer = TokenizerWrapper()

    isolation_model, thresholds = train_isolation_forest(
        prompts=prompts,
        labels=labels,
        tokenizer=tokenizer,
    )

    supervised_path = Path(args.artifact_dir) / "supervised_best.joblib"

    if not supervised_path.exists():
        raise FileNotFoundError(
            "Missing supervised model artifact. Run python -m scripts.train_supervised first."
        )

    supervised_model = joblib.load(supervised_path)

    prompt_sets = create_prompt_length_sets(prompts)

    benchmark_functions = {
        "rule_only": lambda prompt_batch: benchmark_rule_only(prompt_batch),
        "isolation_forest": lambda prompt_batch: benchmark_isolation_only(
            prompt_batch,
            tokenizer,
            isolation_model,
        ),
        "combined_rules_plus_isolation": lambda prompt_batch: benchmark_combined_isolation(
            prompt_batch,
            tokenizer,
            isolation_model,
            thresholds,
        ),
        "supervised_best": lambda prompt_batch: benchmark_supervised(
            prompt_batch,
            tokenizer,
            supervised_model,
        ),
    }

    results: dict[str, dict[str, dict[str, float]]] = {}

    print("\n=== PromptShield Runtime Benchmark ===")
    print(f"Dataset: {args.dataset}")
    print(f"Sample size: {len(prompts)}")

    for variant, benchmark_fn in benchmark_functions.items():
        results[variant] = {}

        for length_group, prompt_batch in prompt_sets.items():
            latencies = benchmark_fn(prompt_batch)
            metrics = summarize_latencies(latencies)
            results[variant][length_group] = metrics

            print(f"\nVariant: {variant}")
            print(f"Prompt length group: {length_group}")
            print(f"Mean latency: {metrics['mean_ms']:.4f} ms")
            print(f"Median latency: {metrics['median_ms']:.4f} ms")
            print(f"P95 latency: {metrics['p95_ms']:.4f} ms")
            print(f"P99 latency: {metrics['p99_ms']:.4f} ms")
            print(
                "Throughput: "
                f"{metrics['throughput_prompts_per_second']:.2f} prompts/sec"
            )

    result_path = Path(args.results_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    json_output = result_path / "runtime_benchmark.json"
    csv_output = result_path / "runtime_benchmark.csv"

    json_output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv_summary(results, str(csv_output))

    print(f"\nWrote runtime benchmark JSON to: {json_output}")
    print(f"Wrote runtime benchmark CSV to: {csv_output}")


if __name__ == "__main__":
    main()