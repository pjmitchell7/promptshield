from __future__ import annotations
def map_risk_band(score: float) -> str:
    # These thresholds are starter thresholds, not final calibrated ones.
    if score >= 0.75:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def combine_scores(anomaly_score: float, rule_count: int) -> dict[str, object]:
    # For now I am keeping the combination logic intentionally simple.
    # The main goal is to make the system end-to-end and inspectable first.
    normalized_anomaly = max(0.0, min(1.0, anomaly_score))

    rule_boost = 0.0
    if rule_count == 1:
        rule_boost = 0.25
    elif rule_count >= 2:
        rule_boost = 0.5

    combined = min(1.0, normalized_anomaly + rule_boost)

    # Stronger rule evidence should be able to force the final result upward.
    if rule_count >= 2:
        combined = max(combined, 0.75)

    return {
        "anomaly_score": normalized_anomaly,
        "rule_count": rule_count,
        "combined_score": combined,
        "risk_band": map_risk_band(combined),
    }