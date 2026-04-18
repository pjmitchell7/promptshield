from __future__ import annotations

from core.context import detect_context_flags


STRONG_RULES = {
    "hidden_prompt_extraction",
    "override_intent",
    "hidden_channel_spoof",
    "structural_spoof",
    "delimiter_spoof",
    "payload_like_pattern",
}

ARTIFACT_RULES = {
    "base64_like_substring",
    "hex_like_substring",
    "zero_width_chars",
}


def anomaly_band(anomaly_score: float, thresholds: dict[str, float]) -> str:
    if anomaly_score >= thresholds["p99"]:
        return "p99"
    if anomaly_score >= thresholds["p95"]:
        return "p95"
    if anomaly_score >= thresholds["p90"]:
        return "p90"
    return "normal"


def map_risk_band(
    suspicious: bool,
    strong_rule_count: int,
    artifact_rule_count: int,
    anomaly_level: str,
) -> str:
    if not suspicious:
        return "low"

    if strong_rule_count >= 2:
        return "high"

    if strong_rule_count >= 1 and anomaly_level in {"p95", "p99"}:
        return "high"

    if artifact_rule_count >= 2 and anomaly_level in {"p95", "p99"}:
        return "high"

    if anomaly_level == "p99":
        return "high"

    return "medium"


def combine_evidence(
    text: str,
    anomaly_score: float,
    rule_hits: list[str],
    thresholds: dict[str, float],
) -> dict[str, object]:
    context = detect_context_flags(text)

    strong_hits = sorted([hit for hit in rule_hits if hit in STRONG_RULES])
    artifact_hits = sorted([hit for hit in rule_hits if hit in ARTIFACT_RULES])

    level = anomaly_band(anomaly_score, thresholds)

    # Strong or artifact rule hits still matter immediately.
    if strong_hits or artifact_hits:
        suspicious = True
    else:
        # This is the main refinement:
        # anomaly-only prompts can be softened if the prompt clearly looks like
        # benign technical debugging/output content and there are no malicious rules.
        if level == "p99":
            suspicious = context["benign_technical_strength"] < 2
        elif level == "p95":
            suspicious = context["benign_technical_strength"] < 1
        else:
            suspicious = False

    risk_band = map_risk_band(
        suspicious=suspicious,
        strong_rule_count=len(strong_hits),
        artifact_rule_count=len(artifact_hits),
        anomaly_level=level,
    )

    return {
        "suspicious": suspicious,
        "risk_band": risk_band,
        "anomaly_score": anomaly_score,
        "anomaly_band": level,
        "strong_rule_hits": strong_hits,
        "artifact_rule_hits": artifact_hits,
        "context_flags": context,
    }