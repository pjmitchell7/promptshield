from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.pipeline import save_pipeline_artifacts, train_isolation_forest_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and save PromptShield detector artifacts."
    )
    parser.add_argument("--benign-file", default="data/prompts/benign.txt")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--encoding-name", default="cl100k_base")
    args = parser.parse_args()

    pipeline = train_isolation_forest_pipeline(
        benign_file=args.benign_file,
        encoding_name=args.encoding_name,
    )

    save_pipeline_artifacts(
        pipeline=pipeline,
        artifact_dir=args.artifact_dir,
    )

    artifact_path = Path(args.artifact_dir)
    print("\n=== PromptShield Training Complete ===")
    print(f"Artifact directory: {artifact_path.resolve()}")
    print(f"Model type: {pipeline.metadata['model_type']}")
    print(f"Training prompts: {pipeline.metadata['num_training_prompts']}")
    print("Thresholds:")
    print(json.dumps(pipeline.thresholds, indent=2))


if __name__ == "__main__":
    main()
