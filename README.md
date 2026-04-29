# Domain-Specific Hallucination Detection in Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/varunchundru/hallucination-detector-deberta)

**CS 593 NLP Final Project | Purdue University | Spring 2026**

Authors: Varun Chundru, Debasmita Biswas

## Overview

This project develops a hallucination detection framework for Large Language Model outputs, combining Natural Language Inference (NLI) with domain-specific fine-tuning. We detect whether LLM-generated responses are **factual** or **hallucinated** given a knowledge context.

### Key Results (HaluEval)

| Model | Accuracy | F1 Score | AUROC |
|-------|----------|----------|-------|
| TF-IDF + Logistic Regression | 0.21 | 0.21 | — |
| Retrieval Similarity (cosine) | 0.49 | 0.66 | 0.38 |
| Zero-Shot NLI (DeBERTa-MNLI) | 0.50 | 0.43 | 0.65 |
| Fine-Tuned DeBERTa-v3-base | 0.91 | 0.92 | 0.98 |
| **MC Dropout (20 passes)** | **0.93** | **0.93** | **0.98** |
| Calibrated DeBERTa | 0.91 | 0.92 | 0.98 |
| Simple Average (DeBERTa + MC) | 0.92 | 0.92 | 0.98 |
| LR Meta-Classifier (all signals) | 0.93 | 0.93 | 0.96 |

MC Dropout improved F1 from **0.91 → 0.93**. Context ablation confirms genuine entailment reasoning (summarization F1 drops 24% without knowledge). DPO training reduced generator hallucination rate by **55.9%**.

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
│   ├── 02c_baseline_tfidf_xgboost.ipynb                        # Debasmita — TF-IDF + XGBoost
│   ├── 03-context-ablation-and-ensemble.ipynb                   # Final — Context ablation + ensemble methods
│   ├── 04-learning-curves.ipynb                                 # Final — Learning curve analysis
│   └── 06_scifact_domain_specific_finetuning.ipynb              # Final — SciFact domain-specific fine-tuning
├── results/
│   ├── update1/
│   │   ├── confusion_matrix_zeroshot.png
│   │   ├── confusion_matrix_finetuned.png
│   │   ├── f1_by_task.png
│   │   ├── results_summary.csv
│   │   └── results_by_task.csv
│   ├── update2/
│   │   ├── calibration_reliability_diagram.png
│   │   ├── retrieval_similarity_distribution.png
│   │   ├── uncertainty_analysis.png
│   │   ├── predictions_tfidf_gradient_boosting.csv
│   │   ├── predictions_tfidf_logistic_regression.csv
│   │   ├── results_tfidf_gradient_boosting.md
│   │   ├── results_tfidf_logistic_regression.md
│   │   ├── scifact_cross_dataset_results.csv
│   │   ├── update2_results_summary.csv
│   │   └── update2_uncertainty_data.csv
│   └── update3/
│       ├── context_ablation.png
│       ├── context_ablation_predictions.csv
│       ├── ensemble_predictions.csv
│       ├── final_notebook03_results.csv
│       ├── learning_curve.png
│       ├── learning_curve_results.csv
│       ├── scifact_test_split.csv
│       └── validation_signals.csv
├── paper/
│   ├── update1/
│   │   ├── acl_latex.tex
│   │   └── custom.bib
│   ├── update2/
│   │   ├── acl_latex_update2.tex
│   │   └── custom.bib
│   └── update3/
│       ├── acl_latex_final.tex
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

### Final: Ablation, Ensemble, DPO & Domain Adaptation

- **Context Ablation:** Stripping knowledge context drops overall F1 from 0.91 → 0.82, confirming the model performs genuine entailment reasoning. Summarization is most context-dependent (F1 drops 24%), while QA is robust (F1 drops only 1%)
- **Ensemble Methods:** Simple averaging (DeBERTa + MC Dropout) achieves F1=0.92; LR meta-classifier combining all four signals (DeBERTa prob, MC mean, MC uncertainty, retrieval similarity) matches MC Dropout at F1=0.93
- **Learning Curves:** DeBERTa trained on 10%/25%/50%/100% of data — sharp elbow at ~5K examples (F1=0.70); 100% data yields F1=0.95. Suggests ~5K labeled examples suffice for usable detection in a new domain
- **DPO Hallucination Mitigation:** Direct Preference Optimization on Qwen2.5-0.5B-Instruct using HaluEval preference pairs reduces generator hallucination rate from 85.5% → 37.7% (55.9% relative reduction), validated by our detector in a closed-loop setup
- **SciFact Domain-Specific Fine-Tuning:** Three configurations evaluated — PubMedBERT fine-tuned on SciFact achieves the best results (Acc=0.76, F1=0.63, AUROC=0.81), demonstrating domain-matched pre-training is the strongest adaptation strategy

## Dataset

We use [HaluEval](https://github.com/RUCAIBox/HaluEval) (Li et al., EMNLP 2023):

| Task | Train | Validation | Test |
|------|-------|------------|------|
| QA | 7,000 | 1,500 | 1,500 |
| Dialogue | 7,000 | 1,500 | 1,500 |
| Summarization | 7,000 | 1,500 | 1,500 |
| **Total** | **21,000** | **4,500** | **4,500** |

## Performance by Task (Fine-Tuned DeBERTa)

| Task | Accuracy | F1 | AUROC |
|------|----------|----|-------|
| QA | 0.97 | 0.97 | 1.00 |
| Summarization | 0.94 | 0.96 | 0.99 |
| Dialogue | 0.83 | 0.82 | 0.94 |

## DPO Hallucination Mitigation

| Model | Hallucination Rate | Mean P(hall) |
|-------|--------------------|--------------|
| Base (Qwen2.5-0.5B-Instruct) | 85.5% | 0.816 |
| DPO-trained | 37.7% | 0.293 |
| **Relative reduction** | **55.9%** | **64.1%** |

## SciFact Domain-Specific Fine-Tuning

| Config | Base Model | Accuracy | F1 | AUROC |
|--------|-----------|----------|-----|-------|
| Zero-shot | HaluEval-DeBERTa | 0.35 | 0.52 | 0.52 |
| A | DeBERTa-v3-base | 0.54 | 0.27 | 0.57 |
| **B** | **PubMedBERT** | **0.76** | **0.63** | **0.81** |
| C | HaluEval→SciFact transfer | 0.47 | 0.53 | 0.61 |

## Roadmap

- [x] **Update 1**: Baseline implementations (Zero-shot NLI + Fine-tuned DeBERTa)
- [x] **Update 2**: Retrieval-augmented verification, MC Dropout uncertainty, temperature scaling, SciFact evaluation
- [x] **Final**: Context ablation, ensemble methods, learning curves, DPO hallucination mitigation, SciFact domain-specific fine-tuning

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
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)
- [Domain-Specific Language Model Pretraining for Biomedical NLP](https://arxiv.org/abs/2007.15779) (Gu et al., 2021)
- [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142) (Gal & Ghahramani, 2016)

## License

MIT License