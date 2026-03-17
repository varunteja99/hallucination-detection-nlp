# Hallucination Detection in LLM Outputs

A semester-long research project for **CS 59300: Natural Language Processing** at Purdue University Fort Wayne (Spring 2026).

## Overview

Large language models frequently generate fluent but factually incorrect text. This project develops and evaluates methods for **domain-specific hallucination detection**, focusing on response-level binary classification (hallucinated vs. faithful) with span-level detection as a stretch goal.

## Approaches

- **NLI-Based Detection** — Using natural language inference models to check entailment between source documents and generated responses
- **Retrieval-Augmented Verification** — Retrieving relevant evidence passages and comparing them against LLM outputs
- **Uncertainty Quantification** — Leveraging model confidence signals (e.g., token-level entropy, sampling consistency) to flag unreliable generations

## Project Structure
```
├── data/               # Datasets and preprocessing scripts
├── src/                # Source code for detection methods
├── experiments/        # Training and evaluation scripts
├── results/            # Output logs and metrics
└── docs/               # Reports and proposal
```

## Datasets

- [HaluEval](https://github.com/RUCAIBox/HaluEval) — Benchmark for hallucination evaluation
- Additional domain-specific datasets TBD

## Team

- **Varun Chundru** — Purdue University
- **Debasmita Biswas** — Purdue University

## References

- HaluEval: A Large-Scale Hallucination Evaluation Benchmark for LLMs
- SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection

## License

MIT
