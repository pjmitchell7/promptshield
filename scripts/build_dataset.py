from __future__ import annotations

import argparse
import csv
from pathlib import Path

from core.pipeline import load_prompts


DATASET_SOURCES = [
    {
        "path": "data/prompts/benign.txt",
        "label": 0,
        "source_split": "benign",
        "category": "benign_general",
    },
    {
        "path": "data/prompts/benign_qna.txt",
        "label": 0,
        "source_split": "benign",
        "category": "benign_qna",
    },
    {
        "path": "data/prompts/benign_writing.txt",
        "label": 0,
        "source_split": "benign",
        "category": "benign_writing",
    },
    {
        "path": "data/prompts/benign_code.txt",
        "label": 0,
        "source_split": "benign",
        "category": "benign_code_debugging",
    },
    {
        "path": "data/prompts/adversarial.txt",
        "label": 1,
        "source_split": "adversarial",
        "category": "direct_prompt_injection",
    },
    {
        "path": "data/prompts/perturbed.txt",
        "label": 1,
        "source_split": "perturbed",
        "category": "perturbed_or_obfuscated",
    },
    {
        "path": "data/prompts/expanded_benign.txt",
        "label": 0,
        "source_split": "expanded_benign",
        "category": "expanded_benign_systems_and_security",
    },
    {
        "path": "data/prompts/expanded_adversarial.txt",
        "label": 1,
        "source_split": "expanded_adversarial",
        "category": "expanded_prompt_injection",
    },
]


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_prompts: set[str] = set()

    for source in DATASET_SOURCES:
        path = Path(str(source["path"]))

        if not path.exists():
            continue

        prompts = load_prompts(str(path))

        for prompt in prompts:
            normalized_key = " ".join(prompt.split()).lower()

            if normalized_key in seen_prompts:
                continue

            seen_prompts.add(normalized_key)

            rows.append(
                {
                    "prompt": prompt,
                    "label": int(source["label"]),
                    "source_split": source["source_split"],
                    "category": source["category"],
                    "source_file": str(path),
                }
            )

    return rows


def write_dataset(rows: list[dict[str, object]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "prompt",
        "label",
        "source_split",
        "category",
        "source_file",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    label_counts: dict[int, int] = {}
    category_counts: dict[str, int] = {}

    for row in rows:
        label = int(row["label"])
        category = str(row["category"])

        label_counts[label] = label_counts.get(label, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1

    print("\n=== Dataset Build Summary ===")
    print(f"Total rows: {len(rows)}")
    print(f"Benign rows: {label_counts.get(0, 0)}")
    print(f"Attack rows: {label_counts.get(1, 0)}")

    print("\nCategory counts:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build normalized PromptShield supervised dataset."
    )
    parser.add_argument(
        "--output",
        default="data/processed/prompts_labeled.csv",
    )
    args = parser.parse_args()

    rows = build_rows()
    write_dataset(rows, args.output)
    print_summary(rows)
    print(f"\nWrote dataset to: {args.output}")


if __name__ == "__main__":
    main()