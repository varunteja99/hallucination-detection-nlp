# Models

## Fine-Tuned Hallucination Detector

Our fine-tuned DeBERTa-v3-base model is hosted on HuggingFace Hub:

🤗 **[varunchundru/hallucination-detector-deberta](https://huggingface.co/varunchundru/hallucination-detector-deberta)**

### Model Details

| Property | Value |
|----------|-------|
| Base Model | microsoft/deberta-v3-base |
| Parameters | 184M |
| Task | Binary Classification (Factual vs Hallucinated) |
| Training Data | HaluEval (21,000 samples) |
| F1 Score | 0.91 (0.93 with MC Dropout) |
| AUROC | 0.98 |

### Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load from HuggingFace
model_name = "varunchundru/hallucination-detector-deberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example inference
text = """Knowledge: Python was created by Guido van Rossum.
Question: Who created Python?
Answer: Python was created by James Gosling."""

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(outputs.logits, dim=-1).item()

label = "Hallucinated" if prediction == 1 else "Factual"
confidence = probs[0][prediction].item()

print(f"Prediction: {label} (confidence: {confidence:.2%})")
# Output: Prediction: Hallucinated (confidence: 94.32%)
```

### MC Dropout Inference (Recommended)

For improved accuracy (F1: 0.91 → 0.93), use MC Dropout with 20 stochastic forward passes:

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "varunchundru/hallucination-detector-deberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = """Knowledge: Python was created by Guido van Rossum.
Question: Who created Python?
Answer: Python was created by James Gosling."""

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

# MC Dropout: enable dropout at inference
model.train()  # keeps dropout active
T = 20  # number of stochastic passes

probs_list = []
with torch.no_grad():
    for _ in range(T):
        outputs = model(**inputs)
        probs_list.append(F.softmax(outputs.logits, dim=-1))

probs = torch.stack(probs_list)
mean_probs = probs.mean(dim=0)
uncertainty = probs[:, :, 1].std(dim=0).item()

prediction = torch.argmax(mean_probs, dim=-1).item()
label = "Hallucinated" if prediction == 1 else "Factual"

print(f"Prediction: {label} | Confidence: {mean_probs[0][prediction]:.2%} | Uncertainty: {uncertainty:.4f}")
# High uncertainty → flag for human review
```

### Label Mapping

| Label ID | Label Name | Description |
|----------|------------|-------------|
| 0 | Factual | Response is supported by the knowledge |
| 1 | Hallucinated | Response contradicts or is unsupported by knowledge |

### Training Configuration

```python
TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=788,
    fp16=False,  # float32 for stability
)
```

## Detection Capabilities

| Capability | Method | Performance |
|------------|--------|-------------|
| Standard inference | Single forward pass | F1=0.91, AUROC=0.98 |
| Uncertainty-aware | MC Dropout (20 passes) | F1=0.93, AUROC=0.98 |
| Calibrated | Temperature scaling | ECE reduced 36.8% |
| Ensemble | LR meta-classifier (4 signals) | F1=0.93, AUROC=0.96 |

### Per-Task Performance

| Task | F1 | AUROC |
|------|----|-------|
| QA | 0.97 | 1.00 |
| Summarization | 0.96 | 0.99 |
| Dialogue | 0.82 | 0.94 |

### Cross-Domain (SciFact)

General-domain training transfers poorly to biomedical claims (F1=0.52). For domain-specific use, see notebook `06_scifact_domain_specific_finetuning.ipynb` — PubMedBERT fine-tuned on SciFact achieves F1=0.63, AUROC=0.81.

---

## DPO-Trained Hallucination Mitigator

Our DPO fine-tuned generator model is hosted on HuggingFace Hub:

🤗 **[varunchundru/dpo-qwen2.5-0.5b-halueval](https://huggingface.co/varunchundru/dpo-qwen2.5-0.5b-halueval)**

### Model Details

| Property | Value |
|----------|-------|
| Base Model | Qwen/Qwen2.5-0.5B-Instruct |
| Parameters | 0.5B |
| Task | Text Generation (hallucination-reduced) |
| Training Method | Direct Preference Optimization (DPO) |
| Training Data | HaluEval (21,000 preference pairs) |
| Hallucination Rate | 85.5% → 37.7% (55.9% relative reduction) |

### Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "varunchundru/dpo-qwen2.5-0.5b-halueval"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

def answer_grounded(question: str, knowledge: str, max_new_tokens: int = 150) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer based only on the provided knowledge. "
                "Do not invent facts. Be concise."
            ),
        },
        {
            "role": "user",
            "content": f"Knowledge: {knowledge}\n\nQuestion: {question}",
        },
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response.strip()

# Example
knowledge = "The Eiffel Tower is located in Paris, France, and was completed in 1889."
question = "Where is the Eiffel Tower located and when was it finished?"
print(answer_grounded(question, knowledge))
```

### Evaluation

Hallucination rates measured by passing generations through `varunchundru/hallucination-detector-deberta` at 0.5 threshold on the held-out test split (4,500 examples).

**Overall:**

| Model | Hallucination Rate | Mean P(hall) |
|-------|--------------------| -------------|
| Base (Qwen2.5-0.5B-Instruct) | 85.5% | 0.816 |
| DPO-trained | 37.7% | 0.293 |
| **Relative reduction** | **55.9%** | **64.1%** |

**Per-Task Breakdown:**

| Task | Base Rate | DPO Rate | Reduction |
|------|-----------|----------|-----------|
| QA | 56.7% | 18.9% | −37.8 pp |
| Summarization | 100% | 0.0% | −100 pp |
| Dialogue | 99.8% | 94.2% | −5.6 pp |

> **Note:** Summarization hallucination is effectively eliminated. Dialogue shows minimal improvement — multi-turn conversation may require more training, a larger model, or task-specific preference data.

### Limitations

- Dialogue performance is weak — DPO did not meaningfully reduce hallucination for multi-turn dialogue (94.2% post-DPO rate)
- Trained for only 1 epoch on 0.5B parameters — further training or a larger base model would likely improve results
- Hallucination rates are measured by a proxy DeBERTa classifier, not human annotation
- `max_length=512` during DPO training may truncate long documents in the summarization task
- Should not be used in high-stakes domains without further validation