from __future__ import annotations
from collections import Counter


# These characters are easy to miss visually, which is part of why they matter.
ZERO_WIDTH_CHARS = {
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\ufeff",  # zero width no-break space / BOM
}

# This is a starter homoglyph list for version 1.
# It is not meant to be perfect yet, just useful enough to catch obvious cases.
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


def count_zero_width(text: str) -> int:
    # Simple count of invisible characters that can distort structure.
    return sum(1 for ch in text if ch in ZERO_WIDTH_CHARS)


def count_non_ascii(text: str) -> int:
    # Non-ASCII is not automatically bad, but it is still worth measuring.
    return sum(1 for ch in text if ord(ch) > 127)


def count_homoglyphs(text: str) -> int:
    # Counts characters that visually resemble safer Latin characters.
    return sum(1 for ch in text if ch in HOMOGLYPH_MAP)


def repeated_token_ratio(tokens: list[int]) -> float:
    # This measures how repetitive the token stream is overall.
    if not tokens:
        return 0.0

    counts = Counter(tokens)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(tokens)


def safe_mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def safe_std(values: list[int]) -> float:
    if not values:
        return 0.0

    mean = safe_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def extract_features(text: str, tokens: list[int]) -> dict[str, float]:
    # These are the locked v1 features for PromptShield.
    # I want the first pass to stay focused instead of ballooning immediately.
    num_tokens = len(tokens)
    text_len = len(text)

    if num_tokens == 0:
        token_id_range = 0.0
        mean_token_id = 0.0
        std_token_id = 0.0
        high_token_ratio = 0.0
    else:
        token_id_range = float(max(tokens) - min(tokens))
        mean_token_id = float(safe_mean(tokens))
        std_token_id = float(safe_std(tokens))

        # This is a rough proxy for whether the prompt is landing in more unusual
        # token ID territory. It may need adjustment later.
        high_token_ratio = sum(1 for t in tokens if t > 100000) / num_tokens

    # This gives a rough sense of how fragmented the prompt becomes after tokenization.
    char_to_token_ratio = (text_len / num_tokens) if num_tokens > 0 else 0.0

    # Again, not inherently suspicious, just something worth tracking.
    non_ascii_ratio = (count_non_ascii(text) / text_len) if text_len > 0 else 0.0

    features = {
        "num_tokens": float(num_tokens),
        "token_id_range": token_id_range,
        "mean_token_id": mean_token_id,
        "std_token_id": std_token_id,
        "high_token_ratio": float(high_token_ratio),
        "repeated_token_ratio": float(repeated_token_ratio(tokens)),
        "char_to_token_ratio": float(char_to_token_ratio),
        "non_ascii_ratio": float(non_ascii_ratio),
        "zero_width_count": float(count_zero_width(text)),
        "homoglyph_count": float(count_homoglyphs(text)),
    }

    return features


def feature_names() -> list[str]:
    # Keeping the order centralized matters because the model needs a stable feature matrix.
    return [
        "num_tokens",
        "token_id_range",
        "mean_token_id",
        "std_token_id",
        "high_token_ratio",
        "repeated_token_ratio",
        "char_to_token_ratio",
        "non_ascii_ratio",
        "zero_width_count",
        "homoglyph_count",
    ]