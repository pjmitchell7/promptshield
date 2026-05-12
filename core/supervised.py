from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.features import extract_features, feature_names
from core.rules import run_rule_checks
from core.scoring import ARTIFACT_RULES, STRONG_RULES
from core.tokenizer import TokenizerWrapper


KNOWN_RULE_IDS = [
    "base64_like_substring",
    "delimiter_spoof",
    "hex_like_substring",
    "hidden_channel_spoof",
    "hidden_prompt_extraction",
    "override_intent",
    "payload_like_pattern",
    "structural_spoof",
    "zero_width_chars",
]


def supervised_feature_names() -> list[str]:
    names = list(feature_names())

    names.extend(
        [
            "rule_count",
            "strong_rule_count",
            "artifact_rule_count",
            "rule_category_count",
        ]
    )

    for rule_id in KNOWN_RULE_IDS:
        names.append(f"rule_hit__{rule_id}")

    return names


def extract_supervised_features(
    prompt: str,
    tokenizer: TokenizerWrapper,
) -> dict[str, float]:
    tokens = tokenizer.encode(prompt)
    base_features = extract_features(prompt, tokens)
    rule_result = run_rule_checks(prompt)
    rule_hits = set(rule_result["rule_hits"])

    strong_rule_count = sum(1 for hit in rule_hits if hit in STRONG_RULES)
    artifact_rule_count = sum(1 for hit in rule_hits if hit in ARTIFACT_RULES)

    features = dict(base_features)
    features["rule_count"] = float(rule_result["rule_count"])
    features["strong_rule_count"] = float(strong_rule_count)
    features["artifact_rule_count"] = float(artifact_rule_count)
    features["rule_category_count"] = float(len(rule_hits))

    for rule_id in KNOWN_RULE_IDS:
        features[f"rule_hit__{rule_id}"] = 1.0 if rule_id in rule_hits else 0.0

    return features


def dicts_to_matrix(feature_dicts: list[dict[str, float]]) -> np.ndarray:
    names = supervised_feature_names()
    rows = []

    for feature_dict in feature_dicts:
        rows.append([feature_dict.get(name, 0.0) for name in names])

    return np.array(rows, dtype=float)


class PromptSupervisedModel:
    def __init__(
        self,
        model_type: str = "logistic_regression",
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.random_state = random_state
        self._is_fit = False
        self._feature_names = supervised_feature_names()
        self.model = self._build_model(model_type=model_type)

    def _build_model(self, model_type: str) -> Any:
        if model_type == "logistic_regression":
            return Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )

        if model_type == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                l2_regularization=0.01,
                random_state=self.random_state,
            )

        raise ValueError(f"Unsupported supervised model type: {model_type}")

    def fit(self, feature_dicts: list[dict[str, float]], labels: list[int]) -> None:
        X = dicts_to_matrix(feature_dicts)
        y = np.array(labels, dtype=int)
        self.model.fit(X, y)
        self._is_fit = True

    def predict(self, feature_dicts: list[dict[str, float]]) -> list[int]:
        if not self._is_fit:
            raise RuntimeError("Supervised model must be fit before prediction.")

        X = dicts_to_matrix(feature_dicts)
        preds = self.model.predict(X)
        return [int(pred) for pred in preds]

    def predict_proba(self, feature_dicts: list[dict[str, float]]) -> list[float]:
        if not self._is_fit:
            raise RuntimeError("Supervised model must be fit before prediction.")

        X = dicts_to_matrix(feature_dicts)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            return [float(prob) for prob in probs[:, 1]]

        scores = self.model.decision_function(X)
        return [float(score) for score in scores]

    def metadata(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "random_state": self.random_state,
            "feature_names": self._feature_names,
        }