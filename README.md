# PromptShield

PromptShield is a tokenizer-aware pre-inference risk detection system for suspicious prompt structure.

Version 1 focuses on this pipeline:

prompt input → tokenization → feature extraction → anomaly detection + rule-based checks → risk score → output

## Version 1 goals

- Detect suspicious prompt structure before inference
- Use token-level structure as the first signal
- Keep the system lightweight and modular
- Support both a user-facing HCC interpretation and an ML Systems evaluation interpretation

## Current implementation scope

- `tiktoken` tokenizer wrapper
- token-level feature extraction
- Isolation Forest anomaly detector
- rule-based detector
- combined risk scoring
- baseline evaluation harness