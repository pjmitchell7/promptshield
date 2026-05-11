from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from core.features import extract_features, feature_names
from core.model import PromptAnomalyModel
from core.rules import run_rule_checks
from core.scoring import combine_evidence
from core.tokenizer import TokenizerWrapper


class PromptShieldPipeline:
    def __init__(
        self,
        model: PromptAnomalyModel,
        tokenizer: TokenizerWrapper,
        thresholds: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.thresholds = thresholds
        self.metadata = metadata or {}

    def score_prompt(self, prompt: str) -> dict[str, object]:
        tokens = self.tokenizer.encode(prompt)
        feature_dict = extract_features(prompt, tokens)
        anomaly_score = self.model.anomaly_score(feature_dict)
        rules = run_rule_checks(prompt)

        evidence = combine_evidence(
            text=prompt,
            anomaly_score=anomaly_score,
            rule_hits=rules["rule_hits"],
            thresholds=self.thresholds,
        )

        return {
            "prompt": prompt,
            "tokens": tokens,
            "features": feature_dict,
            "rules": rules,
            "scoring": evidence,
            "thresholds": self.thresholds,
            "metadata": self.metadata,
        }

    def score_prompts(self, prompts: list[str]) -> list[dict[str, object]]:
        return [self.score_prompt(prompt) for prompt in prompts]


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


def train_isolation_forest_pipeline(
    benign_file: str = "data/prompts/benign.txt",
    encoding_name: str = "cl100k_base",
) -> PromptShieldPipeline:
    tokenizer = TokenizerWrapper(encoding_name=encoding_name)
    benign_prompts = load_prompts(benign_file)
    benign_features = build_feature_dicts(benign_prompts, tokenizer)

    model = PromptAnomalyModel()
    model.fit(benign_features)
    thresholds = model.calibrate_thresholds(benign_features)

    metadata = {
        "model_type": "isolation_forest",
        "encoding_name": encoding_name,
        "benign_file": benign_file,
        "num_training_prompts": len(benign_prompts),
        "feature_names": feature_names(),
    }

    return PromptShieldPipeline(
        model=model,
        tokenizer=tokenizer,
        thresholds=thresholds,
        metadata=metadata,
    )


def save_pipeline_artifacts(
    pipeline: PromptShieldPipeline,
    artifact_dir: str = "artifacts",
) -> None:
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline.model, artifact_path / "isolation_forest.joblib")

    (artifact_path / "thresholds.json").write_text(
        json.dumps(pipeline.thresholds, indent=2),
        encoding="utf-8",
    )

    (artifact_path / "metadata.json").write_text(
        json.dumps(pipeline.metadata, indent=2),
        encoding="utf-8",
    )

    (artifact_path / "feature_schema.json").write_text(
        json.dumps({"feature_names": feature_names()}, indent=2),
        encoding="utf-8",
    )


def load_pipeline_artifacts(
    artifact_dir: str = "artifacts",
) -> PromptShieldPipeline:
    artifact_path = Path(artifact_dir)

    model_path = artifact_path / "isolation_forest.joblib"
    thresholds_path = artifact_path / "thresholds.json"
    metadata_path = artifact_path / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {model_path}. Run scripts/train_detector.py first."
        )

    model = joblib.load(model_path)
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {}

    encoding_name = metadata.get("encoding_name", "cl100k_base")
    tokenizer = TokenizerWrapper(encoding_name=encoding_name)

    return PromptShieldPipeline(
        model=model,
        tokenizer=tokenizer,
        thresholds=thresholds,
        metadata=metadata,
    )
