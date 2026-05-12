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
from core.rules import run_rule_checks
from core.scoring import combine_evidence
from core.supervised import (
    PromptSupervisedModel,
    dicts_to_matrix,
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


def summarize_batch_latencies(
    batch_latencies_ms: list[float],
    total_prompts: int,
) -> dict[str, float]:
    total_ms = sum(batch_latencies_ms)
    total_seconds = total_ms / 1000.0

    if total_seconds <= 0:
        throughput = 0.0
        mean_per_prompt_ms = 0.0
    else:
        throughput = total_prompts / total_seconds
        mean_per_prompt_ms = total_ms / total_prompts

    return {
        "num_batches": float(len(batch_latencies_ms)),
        "total_prompts": float(total_prompts),
        "mean_batch_ms": float(statistics.mean(batch_latencies_ms)),
        "median_batch_ms": float(statistics.median(batch_latencies_ms)),
        "p95_batch_ms": percentile(batch_latencies_ms, 95),
        "p99_batch_ms": percentile(batch_latencies_ms, 99),
        "mean_per_prompt_ms": float(mean_per_prompt_ms),
        "throughput_prompts_per_second": float(throughput),
    }


def chunked(items: list[str], batch_size: int) -> list[list[str]]:
    return [
        items[i : i + batch_size]
        for i in range(0, len(items), batch_size)
        if items[i : i + batch_size]
    ]


def create_prompt_length_sets(base_prompts: list[str]) -> dict[str, list[str]]:
    short_prompts = []
    long_prompts = []

    for prompt in base_prompts:
        one_line = " ".join(prompt.split())
        short_prompts.append(one_line[:200])
        long_prompts.append((one_line + " ") * 16)

    return {
        "short": short_prompts,
        "long": long_prompts,
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


def benchmark_rule_only_batch(
    prompt_batches: list[list[str]],
) -> tuple[list[float], int]:
    batch_latencies = []
    total_prompts = 0

    for batch in prompt_batches:
        start = time.perf_counter()

        for prompt in batch:
            run_rule_checks(prompt)

        end = time.perf_counter()

        total_prompts += len(batch)
        batch_latencies.append((end - start) * 1000.0)

    return batch_latencies, total_prompts


def benchmark_combined_isolation_batch(
    prompt_batches: list[list[str]],
    tokenizer: TokenizerWrapper,
    model: PromptAnomalyModel,
    thresholds: dict[str, float],
) -> tuple[list[float], int]:
    batch_latencies = []
    total_prompts = 0

    for batch in prompt_batches:
        start = time.perf_counter()

        for prompt in batch:
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

        total_prompts += len(batch)
        batch_latencies.append((end - start) * 1000.0)

    return batch_latencies, total_prompts


def benchmark_supervised_naive_batch(
    prompt_batches: list[list[str]],
    tokenizer: TokenizerWrapper,
    model: PromptSupervisedModel,
) -> tuple[list[float], int]:
    batch_latencies = []
    total_prompts = 0

    for batch in prompt_batches:
        start = time.perf_counter()

        for prompt in batch:
            features = extract_supervised_features(prompt, tokenizer)
            model.predict([features])

        end = time.perf_counter()

        total_prompts += len(batch)
        batch_latencies.append((end - start) * 1000.0)

    return batch_latencies, total_prompts


def benchmark_supervised_vectorized_batch(
    prompt_batches: list[list[str]],
    tokenizer: TokenizerWrapper,
    model: PromptSupervisedModel,
) -> tuple[list[float], int]:
    batch_latencies = []
    total_prompts = 0

    for batch in prompt_batches:
        start = time.perf_counter()

        feature_dicts = [
            extract_supervised_features(prompt, tokenizer)
            for prompt in batch
        ]
        X = dicts_to_matrix(feature_dicts)
        model.model.predict(X)

        end = time.perf_counter()

        total_prompts += len(batch)
        batch_latencies.append((end - start) * 1000.0)

    return batch_latencies, total_prompts


def write_csv_summary(
    results: list[dict[str, object]],
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "variant",
        "prompt_length_group",
        "batch_size",
        "num_batches",
        "total_prompts",
        "mean_batch_ms",
        "median_batch_ms",
        "p95_batch_ms",
        "p99_batch_ms",
        "mean_per_prompt_ms",
        "throughput_prompts_per_second",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PromptShield batch runtime behavior."
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
        default=256,
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
    batch_sizes = [1, 8, 32, 64]

    benchmark_variants = {
        "rule_only": lambda batches: benchmark_rule_only_batch(batches),
        "combined_rules_plus_isolation": lambda batches: benchmark_combined_isolation_batch(
            batches,
            tokenizer,
            isolation_model,
            thresholds,
        ),
        "supervised_naive": lambda batches: benchmark_supervised_naive_batch(
            batches,
            tokenizer,
            supervised_model,
        ),
        "supervised_vectorized": lambda batches: benchmark_supervised_vectorized_batch(
            batches,
            tokenizer,
            supervised_model,
        ),
    }

    results: list[dict[str, object]] = []

    print("\n=== PromptShield Batch Runtime Benchmark ===")
    print(f"Dataset: {args.dataset}")
    print(f"Sample size: {len(prompts)}")

    for length_group, prompt_list in prompt_sets.items():
        print(f"\n--- Prompt length group: {length_group} ---")

        for batch_size in batch_sizes:
            batches = chunked(prompt_list, batch_size=batch_size)

            for variant, benchmark_fn in benchmark_variants.items():
                latencies, total_prompts = benchmark_fn(batches)
                metrics = summarize_batch_latencies(
                    batch_latencies_ms=latencies,
                    total_prompts=total_prompts,
                )

                row = {
                    "variant": variant,
                    "prompt_length_group": length_group,
                    "batch_size": batch_size,
                    **metrics,
                }
                results.append(row)

                print(
                    f"{variant} | batch={batch_size} | "
                    f"mean_per_prompt={metrics['mean_per_prompt_ms']:.4f} ms | "
                    f"throughput={metrics['throughput_prompts_per_second']:.2f} prompts/sec"
                )

    result_path = Path(args.results_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    json_output = result_path / "batch_runtime_benchmark.json"
    csv_output = result_path / "batch_runtime_benchmark.csv"

    json_output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv_summary(results, str(csv_output))

    print(f"\nWrote batch runtime JSON to: {json_output}")
    print(f"Wrote batch runtime CSV to: {csv_output}")


if __name__ == "__main__":
    main()