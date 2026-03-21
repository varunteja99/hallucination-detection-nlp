# Domain-Specific Hallucination Detection in Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/varunchundru/hallucination-detector-deberta)

**CS 593 NLP Final Project | Purdue University | Spring 2026**

Authors: Varun Chundru, Debasmita Biswas

## Overview

This project develops a hallucination detection framework for Large Language Model outputs, combining Natural Language Inference (NLI) with domain-specific fine-tuning. We detect whether LLM-generated responses are **factual** or **hallucinated** given a knowledge context.

### Key Results (Update 1)

| Model | Accuracy | Precision | Recall | F1 Score | AUROC |
|-------|----------|-----------|--------|----------|-------|
| Zero-Shot NLI (DeBERTa-MNLI) | 0.60 | 0.73 | 0.31 | 0.43 | 0.65 |
| **Fine-Tuned DeBERTa-v3-base** | **0.91** | **0.88** | **0.95** | **0.91** | **0.98** |

Fine-tuning improved F1 by **111.9%** over the zero-shot baseline.

## Project Structure

```
hallucination-detection-nlp/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_baseline_nli_finetuning.ipynb    # Update 1: Baselines
│   ├── 02_retrieval_augmented.ipynb        # Update 2: Retrieval (planned)
│   └── 03_uncertainty_quantification.ipynb # Update 2: Uncertainty (planned)
├── results/
│   └── update1/
│       ├── confusion_matrix_zeroshot.png
│       ├── confusion_matrix_finetuned.png
│       ├── f1_by_task.png
│       ├── results_summary.csv
│       └── results_by_task.csv
├── paper/
│   └── update1/
│       ├── acl_latex_update1.tex
│       └── custom.bib
└── models/
    └── README.md                           # Links to HuggingFace
```

## Quick Start

### Installation

```bash
git clone https://github.com/varunteja99/hallucination-detection-nlp.git
cd hallucination-detection-nlp
pip install -r requirements.txt
```

### Using the Fine-Tuned Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model from HuggingFace
model_name = "varunchundru/hallucination-detector-deberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Prepare input (knowledge + question + answer)
text = """Knowledge: The Eiffel Tower is located in Paris, France.
Question: Where is the Eiffel Tower?
Answer: The Eiffel Tower is in London."""

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()

print("Hallucinated" if prediction == 1 else "Factual")
# Output: Hallucinated
```

## Methodology

### Approach 1: Zero-Shot NLI Detection
- Uses DeBERTa-v3-base fine-tuned on MNLI/FEVER/ANLI
- Maps NLI labels to hallucination: Contradiction → Hallucinated, Entailment/Neutral → Factual
- No task-specific training required

### Approach 2: Fine-Tuned Detection (Primary)
- Fine-tunes DeBERTa-v3-base directly for binary hallucination classification
- Trained on HaluEval dataset (21,000 training samples)
- 3 epochs, learning rate 2e-5, batch size 8

## Dataset

We use [HaluEval](https://github.com/RUCAIBox/HaluEval) (Li et al., EMNLP 2023):

| Task | Train | Validation | Test |
|------|-------|------------|------|
| QA | 7,000 | 1,500 | 1,500 |
| Dialogue | 7,000 | 1,500 | 1,500 |
| Summarization | 7,000 | 1,500 | 1,500 |
| **Total** | **21,000** | **4,500** | **4,500** |

## Performance by Task

| Task | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| QA | 0.97 | 0.98 | 0.96 | 0.97 |
| Dialogue | 0.82 | 0.75 | 0.91 | 0.82 |
| Summarization | 0.96 | 0.93 | 0.98 | 0.96 |

## Roadmap

- [x] **Update 1**: Baseline implementations (Zero-shot NLI + Fine-tuned DeBERTa)
- [ ] **Update 2**: Retrieval-augmented verification, Uncertainty quantification
- [ ] **Final**: Domain-specific evaluation (SciFact), Ensemble methods

## Citation

```bibtex
@misc{chundru2026hallucination,
  author = {Chundru, Varun and Biswas, Debasmita},
  title = {Domain-Specific Hallucination Detection in Large Language Models},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/varunteja99/hallucination-detection-nlp}
}
```

## References

- [HaluEval: A Large-Scale Hallucination Evaluation Benchmark](https://arxiv.org/abs/2305.11747) (Li et al., 2023)
- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) (He et al., 2021)

## License

MIT License