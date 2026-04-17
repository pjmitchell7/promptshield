from __future__ import annotations
import re

# Some suspicious patterns are easier to catch with direct rules than with the anomaly model.
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")
BASE64_LIKE_PATTERN = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")
HEX_LIKE_PATTERN = re.compile(r"\b(?:0x)?[A-Fa-f0-9]{16,}\b")

# These are starter phrase-level checks for obvious jailbreak-style wording.
JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?prior\s+rules",
    r"bypass\s+safety",
    r"reveal\s+system\s+prompt",
    r"developer\s+mode",
    r"do\s+not\s+follow\s+your\s+safety\s+policies",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
]

# These are not the main focus of PromptShield, but they are still useful sanity checks.
PAYLOAD_PATTERNS = [
    r"<script.*?>.*?</script>",
    r"javascript:",
    r"data:text/html",
    r"union\s+select",
    r"drop\s+table",
    r"rm\s+-rf",
]


def run_rule_checks(text: str) -> dict[str, object]:
    lowered = text.lower()
    hits: list[str] = []

    if ZERO_WIDTH_PATTERN.search(text):
        hits.append("zero_width_chars")

    if BASE64_LIKE_PATTERN.search(text):
        hits.append("base64_like_substring")

    if HEX_LIKE_PATTERN.search(text):
        hits.append("hex_like_substring")

    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, lowered):
            hits.append("jailbreak_phrase")

    for pattern in PAYLOAD_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE | re.DOTALL):
            hits.append("payload_like_pattern")

    # I only want distinct rule hits here, not duplicates from multiple regex matches.
    unique_hits = sorted(set(hits))

    return {
        "rule_hits": unique_hits,
        "rule_count": len(unique_hits),
    }