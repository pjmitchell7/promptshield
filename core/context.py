from __future__ import annotations

import re


ANALYSIS_CONTEXT_PATTERNS = [
    r"\bas text only\b",
    r"\bquoted example\b",
    r"\bliteral example\b",
    r"\bnot as an instruction\b",
    r"\bdo not obey it\b",
    r"\bexplain why .* could be risky\b",
    r"\blooks suspicious\b",
]

# These patterns are meant to explain away anomaly spikes on clearly benign
# technical prompts. I am keeping them narrow on purpose.
BENIGN_TECHNICAL_PATTERN_GROUPS = {
    "traceback": [
        r"traceback \(most recent call last\):",
        r'file ".*?", line \d+',
    ],
    "exception_names": [
        r"\bmodulenotfounderror\b",
        r"\btypeerror\b",
        r"\bindexerror\b",
        r"\bvalueerror\b",
        r"\bkeyerror\b",
        r"\byaml\.parser\.parsererror\b",
    ],
    "shell_or_cli": [
        r"^\$ ",
        r"\bchmod\s+\d{3}\b",
        r"\bgrep\s+-r\b",
        r"\bssh\s+\S+@\S+\s+-p\s+\d+\b",
        r"\bgit\s+push\s+origin\b",
        r"failed to push some refs",
        r"\bdocker\s+ps\s+-a\b",
    ],
    "file_listing": [
        r"\bls\s+-l\b",
        r"^[\-dl][rwx\-]{9}\s+\d+\s+\S+\s+\S+\s+\d+",
    ],
    "config_debugging": [
        r"\bnetplan\.yaml\b",
        r"\byaml\.parser\b",
    ],
    "code_or_markup_snippet": [
        r"#include\s*<stdio\.h>",
        r"<link\s+rel\s*=\s*[\"']stylesheet[\"']",
        r"\bselect\s+\w+.*,?\s*count\s*\(\*\)",
    ],
}


def normalize_for_context(text: str) -> str:
    # This keeps matching stable across multiline pasted prompts.
    return " ".join(text.lower().split())


def detect_context_flags(text: str) -> dict[str, object]:
    normalized = normalize_for_context(text)
    matched_groups: list[str] = []

    for group_name, patterns in BENIGN_TECHNICAL_PATTERN_GROUPS.items():
        if any(re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL) for pattern in patterns):
            matched_groups.append(group_name)

    analysis_context = any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in ANALYSIS_CONTEXT_PATTERNS
    )

    benign_technical_strength = len(matched_groups)

    return {
        "analysis_context": analysis_context,
        "benign_technical_context": benign_technical_strength > 0,
        "benign_technical_strength": benign_technical_strength,
        "matched_benign_technical_groups": matched_groups,
    }