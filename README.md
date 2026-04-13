# Domain-Specific Hallucination Detection in Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/varunchundru/hallucination-detector-deberta)

**CS 593 NLP Final Project | Purdue University | Spring 2026**

Authors: Varun Chundru, Debasmita Biswas

## Overview

This project develops a hallucination detection framework for Large Language Model outputs, combining Natural Language Inference (NLI) with domain-specific fine-tuning. We detect whether LLM-generated responses are **factual** or **hallucinated** given a knowledge context.

### Key Results

| Model | Accuracy | Precision | Recall | F1 Score | AUROC |
|-------|----------|-----------|--------|----------|-------|
| TF-IDF + Logistic Regression | 0.21 | 0.21 | 0.22 | 0.21 | — |
| Retrieval Similarity (cosine) | 0.49 | 0.49 | 1.00 | 0.66 | 0.38 |
| Zero-Shot NLI (DeBERTa-MNLI) | 0.60 | 0.73 | 0.31 | 0.43 | 0.65 |
| Fine-Tuned DeBERTa-v3-base | 0.91 | 0.88 | 0.95 | 0.91 | 0.98 |
| **MC Dropout (20 passes)** | **0.93** | **0.93** | **0.93** | **0.93** | **0.98** |
| Calibrated DeBERTa | 0.91 | 0.88 | 0.95 | 0.91 | 0.98 |

MC Dropout averaging improved F1 from **0.91 → 0.93**. Incorrect predictions show **5.75× higher uncertainty** than correct ones.

## Project Structure

```
hallucination-detection-nlp/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_baseline_nli_finetuning.ipynb                        # Varun — Update 1: Zero-shot + Fine-tuned DeBERTa
│   ├── 01b_baseline_tfidf_fever.ipynb                          # Debasmita — FEVER baseline
│   ├── 02_update2_retrieval_augmented_verification.ipynb        # Varun — Update 2: Retrieval, MC Dropout, SciFact
│   ├── 02b_tfidf_baseline_logistic_regression.ipynb             # Debasmita — TF-IDF + LogReg on HaluEval
│   └── 02c_baseline_tfidf_xgboost.ipynb                        # Debasmita — TF-IDF + XGBoost
├── results/
│   ├── update1/
│   │   ├── confusion_matrix_zeroshot.png
│   │   ├── confusion_matrix_finetuned.png
│   │   ├── f1_by_task.png
│   │   ├── results_summary.csv
│   │   └── results_by_task.csv
│   └── update2/
│       ├── calibration_reliability_diagram.png
│       ├── retrieval_similarity_distribution.png
│       ├── uncertainty_analysis.png
│       ├── predictions_tfidf_gradient_boosting.csv
│       ├── predictions_tfidf_logistic_regression.csv
│       ├── results_tfidf_gradient_boosting.md
│       ├── results_tfidf_logistic_regression.md
│       ├── scifact_cross_dataset_results.csv
│       ├── update2_results_summary.csv
│       └── update2_uncertainty_data.csv
├── paper/
│   ├── update1/
│   │   ├── acl_latex.tex
│   │   └── custom.bib
│   └── update2/
│       ├── acl_latex_update2.tex
│       └── custom.bib
└── models/
    └── README.md                                                # Links to HuggingFace
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

### Update 1: Baseline Detection

- **Zero-Shot NLI:** DeBERTa-v3-base pre-trained on MNLI/FEVER/ANLI, maps contradiction → hallucinated
- **Fine-Tuned DeBERTa:** Binary classifier fine-tuned on HaluEval (21,000 samples, 3 epochs, lr=2e-5)

### Update 2: Advanced Detection & Analysis

- **TF-IDF + Logistic Regression:** Simple ML baseline confirming lexical features are insufficient (21% accuracy, below random chance)
- **Retrieval-Augmented Verification:** Dense cosine similarity between knowledge and response using sentence-transformers; insufficient as standalone detector (F1=0.66)
- **MC Dropout Uncertainty:** 20 stochastic forward passes to estimate prediction uncertainty; incorrect predictions show 5.75× higher uncertainty; selective prediction achieves ~100% accuracy at 5% coverage
- **Temperature Scaling:** Reduces Expected Calibration Error from 0.078 → 0.050 (36.8% improvement)
- **SciFact Cross-Dataset Evaluation:** PubMedBERT retrieval + DeBERTa classification on scientific claims reveals significant domain gap (F1 drops 0.91 → 0.50)

## Dataset

We use [HaluEval](https://github.com/RUCAIBox/HaluEval) (Li et al., EMNLP 2023):

| Task | Train | Validation | Test |
|------|-------|------------|------|
| QA | 7,000 | 1,500 | 1,500 |
| Dialogue | 7,000 | 1,500 | 1,500 |
| Summarization | 7,000 | 1,500 | 1,500 |
| **Total** | **21,000** | **4,500** | **4,500** |

## Performance by Task (Fine-Tuned DeBERTa)

| Task | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| QA | 0.97 | 0.98 | 0.96 | 0.97 |
| Dialogue | 0.82 | 0.75 | 0.91 | 0.82 |
| Summarization | 0.96 | 0.93 | 0.98 | 0.96 |

## Roadmap

- [x] **Update 1**: Baseline implementations (Zero-shot NLI + Fine-tuned DeBERTa)
- [x] **Update 2**: Retrieval-augmented verification, MC Dropout uncertainty, temperature scaling, SciFact evaluation
- [ ] **Final**: Ensemble methods, context ablation, learning curves, presentation

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
- [Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974) (Wadden et al., 2020)
- [TRUE: Re-evaluating Factual Consistency Evaluation](https://arxiv.org/abs/2204.04991) (Honovich et al., 2022)

## License

MIT License