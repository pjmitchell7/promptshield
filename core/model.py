from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from core.features import feature_names


class PromptAnomalyModel:
    # Isolation Forest is the locked v1 anomaly model.
    # It is lightweight, unsupervised, and enough to get the backbone working.
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
        # This converts feature dictionaries into a stable numeric matrix.
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

        # decision_function gives higher values for more normal points.
        # I am flipping the sign so higher means more suspicious.
        raw_score = self.model.decision_function(X)[0]

        return float(-raw_score)

    def save_metadata(self, output_path: str) -> None:
        # This is just a small helper so the run setup can be documented later.
        metadata = {
            "feature_names": self._feature_names,
            "random_state": self.random_state,
        }
        Path(output_path).write_text(json.dumps(metadata, indent=2))