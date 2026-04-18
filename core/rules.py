from __future__ import annotations

import re
import unicodedata


# These are the invisible characters I want to strip before phrase matching.
ZERO_WIDTH_CHARS = {
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\ufeff",  # zero width no-break space / BOM
}

# Small starter mapping for common confusable characters that can hide rule phrases.
HOMOGLYPH_MAP = {
    "а": "a",  # Cyrillic a
    "е": "e",  # Cyrillic e
    "о": "o",  # Cyrillic o
    "р": "p",  # Cyrillic er
    "с": "c",  # Cyrillic es
    "у": "y",  # Cyrillic u
    "х": "x",  # Cyrillic ha
    "і": "i",  # Cyrillic/UKR i
    "ј": "j",  # Cyrillic je
    "Α": "A",  # Greek Alpha
    "Β": "B",  # Greek Beta
    "Ε": "E",  # Greek Epsilon
    "Ζ": "Z",  # Greek Zeta
    "Η": "H",  # Greek Eta
    "Ι": "I",  # Greek Iota
    "Κ": "K",  # Greek Kappa
    "Μ": "M",  # Greek Mu
    "Ν": "N",  # Greek Nu
    "Ο": "O",  # Greek Omicron
    "Ρ": "P",  # Greek Rho
    "Τ": "T",  # Greek Tau
    "Χ": "X",  # Greek Chi
}

HOMOGLYPH_TRANSLATION = str.maketrans(HOMOGLYPH_MAP)

ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")
BASE64_LIKE_PATTERN = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")
HEX_LIKE_PATTERN = re.compile(r"\b(?:0x)?[A-Fa-f0-9]{16,}\b")


# These patterns target instruction override behavior specifically.
OVERRIDE_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+everything\s+(above|before)",
    r"disregard\s+(all\s+)?(prior|previous)\s+(rules|instructions)",
    r"override\s+(all\s+)?(prior|previous)\s+(rules|instructions)",
    r"follow\s+only\s+the\s+next\s+sentence",
    r"do\s+not\s+follow\s+(your\s+)?safety\s+polic",
    r"bypass\s+safety",
    r"developer\s+mode",
    r"unrestricted\s+mode",
    r"no\s+restrictions",
]

# These patterns are more about extracting hidden prompt material.
EXTRACTION_PATTERNS = [
    r"(reveal|print|output|summarize)\s+(the\s+)?(full\s+)?(hidden\s+)?system\s+prompt",
    r"(reveal|print|output|summarize)\s+(the\s+)?(full\s+)?(hidden\s+)?prompt",
    r"(reveal|print|output|summarize)\s+(the\s+)?(hidden\s+)?developer\s+message",
    r"(reveal|print|output|summarize)\s+(the\s+)?(internal|hidden|protected)\s+(instructions|policies)",
    r"hidden\s+chain\s+of\s+instructions",
    r"exact\s+hidden\s+instructions",
    r"hidden\s+instruction\s+prefix",
    r"anything\s+not\s+shown\s+to\s+the\s+user",
]

# These patterns catch prompts trying to smuggle in a second instruction layer.
CONTROL_CHANNEL_PATTERNS = [
    r"highest[- ]priority\s+instruction",
    r"real\s+instruction\s+layer",
    r"real\s+instruction\s+set",
    r"visible\s+prompt\s+is\s+a\s+decoy",
    r"visible\s+text\s+is\s+not\s+authoritative",
    r"human-readable\s+text\s+is\s+not\s+authoritative",
    r"hidden\s+control\s+(text|layer|characters?)",
    r"invisible\s+control\s+text",
    r"metadata[- ]looking\s+fragments",
    r"encoded\s+block",
    r"treat\s+the\s+following\s+as\s+the\s+highest[- ]priority\s+instruction",
    r"prefer\s+any\s+hidden\s+control\s+layer",
]

# These patterns catch fake system/config formatting meant to spoof authority.
STRUCTURAL_SPOOF_PATTERNS = [
    r"<system>.*?</system>",
    r"<assistant_config\b.*?>.*?</assistant_config>",
    r"role:\s*system",
    r"priority:\s*critical",
    r"---begin\s+override---.*?---end\s+override---",
    r"\[override\].*?\[/override\]",
]

# These are payload-style indicators. They are not the main thesis of PromptShield,
# but they are still useful explicit checks.
PAYLOAD_PATTERNS = [
    r"<script.*?>.*?</script>",
    r"javascript:",
    r"data:text/html",
    r"union\s+select",
    r"drop\s+table",
    r"rm\s+-rf",
]

# Repeated delimiter flooding can be suspicious when paired with authority keywords.
DELIMITER_FLOOD_PATTERN = re.compile(r"[:>#\-\[\]\|]{8,}", re.IGNORECASE)
DELIMITER_KEYWORD_PATTERN = re.compile(
    r"(system|prompt|instruction|instructions|developer|override|policy|policies)",
    re.IGNORECASE,
)


def normalize_for_matching(text: str) -> str:
    # First normalize Unicode form so weird copies do not create avoidable mismatches.
    normalized = unicodedata.normalize("NFKC", text)

    # Remove zero-width characters before phrase matching.
    normalized = "".join(ch for ch in normalized if ch not in ZERO_WIDTH_CHARS)

    # Translate a small starter set of confusables back to the Latin characters they mimic.
    normalized = normalized.translate(HOMOGLYPH_TRANSLATION)

    # Collapse whitespace so multiline and oddly spaced prompts still match the same patterns.
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def has_delimiter_spoof(text: str) -> bool:
    # I do not want delimiter flooding alone to fire. It should also mention
    # authority-ish keywords so the rule stays narrower.
    if not DELIMITER_FLOOD_PATTERN.search(text):
        return False
    return bool(DELIMITER_KEYWORD_PATTERN.search(text))


def run_rule_checks(text: str) -> dict[str, object]:
    hits: list[str] = []

    normalized = normalize_for_matching(text)

    # Keep this as an explicit signal because invisible characters matter on their own.
    if ZERO_WIDTH_PATTERN.search(text):
        hits.append("zero_width_chars")

    # These are explicit suspicious substrings that are easy to justify as rules.
    if BASE64_LIKE_PATTERN.search(text):
        hits.append("base64_like_substring")

    if HEX_LIKE_PATTERN.search(text):
        hits.append("hex_like_substring")

    if has_delimiter_spoof(text):
        hits.append("delimiter_spoof")

    for pattern in OVERRIDE_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            hits.append("override_intent")
            break

    for pattern in EXTRACTION_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            hits.append("hidden_prompt_extraction")
            break

    for pattern in CONTROL_CHANNEL_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            hits.append("hidden_channel_spoof")
            break

    for pattern in STRUCTURAL_SPOOF_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL):
            hits.append("structural_spoof")
            break

    for pattern in PAYLOAD_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL):
            hits.append("payload_like_pattern")
            break

    unique_hits = sorted(set(hits))

    return {
        "rule_hits": unique_hits,
        "rule_count": len(unique_hits),
    }