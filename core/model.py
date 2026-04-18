from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from core.features import feature_names


class PromptAnomalyModel:
    # Isolation Forest is still the right v1 model.
    # The problem now is not model choice. It is better calibration and arbitration.
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=random_state,
        )
        self._feature_names = feature_names()
        self._is_fit = False

    def dicts_to_matrix(self, feature_dicts: list[dict[str, float]]) -> np.ndarray:
        rows = []
        for feature_dict in feature_dicts:
            rows.append([feature_dict[name] for name in self._feature_names])
        return np.array(rows, dtype=float)

    def fit(self, feature_dicts: list[dict[str, float]]) -> None:
        X = self.dicts_to_matrix(feature_dicts)
        self.model.fit(X)
        self._is_fit = True

    def anomaly_score(self, feature_dict: dict[str, float]) -> float:
        if not self._is_fit:
            raise RuntimeError("Model must be fit before scoring.")

        X = self.dicts_to_matrix([feature_dict])

        # sklearn's decision_function is higher for more normal points.
        # I flip the sign so higher means more suspicious in PromptShield terms.
        raw_score = self.model.decision_function(X)[0]
        return float(-raw_score)

    def anomaly_scores(self, feature_dicts: list[dict[str, float]]) -> list[float]:
        if not self._is_fit:
            raise RuntimeError("Model must be fit before scoring.")

        X = self.dicts_to_matrix(feature_dicts)
        raw_scores = self.model.decision_function(X)
        return [float(-score) for score in raw_scores]

    def calibrate_thresholds(self, benign_feature_dicts: list[dict[str, float]]) -> dict[str, float]:
        scores = np.array(self.anomaly_scores(benign_feature_dicts), dtype=float)

        return {
            "p90": float(np.percentile(scores, 90)),
            "p95": float(np.percentile(scores, 95)),
            "p99": float(np.percentile(scores, 99)),
        }

    def save_metadata(self, output_path: str) -> None:
        metadata = {
            "feature_names": self._feature_names,
            "random_state": self.random_state,
        }
        Path(output_path).write_text(json.dumps(metadata, indent=2))