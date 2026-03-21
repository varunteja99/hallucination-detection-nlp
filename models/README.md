# Models

## Fine-Tuned Hallucination Detector

Our fine-tuned DeBERTa-v3-base model is hosted on HuggingFace Hub:

🤗 **[varunteja99/hallucination-detector-deberta](https://huggingface.co/varunteja99/hallucination-detector-deberta)**

### Model Details

| Property | Value |
|----------|-------|
| Base Model | microsoft/deberta-v3-base |
| Parameters | 184M |
| Task | Binary Classification (Factual vs Hallucinated) |
| Training Data | HaluEval (21,000 samples) |
| F1 Score | 0.91 |
| AUROC | 0.98 |

### Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load from HuggingFace
model_name = "varunteja99/hallucination-detector-deberta"
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

## Future Models

- [ ] `hallucination-detector-deberta-retrieval` - With retrieval augmentation (Update 2)
- [ ] `hallucination-detector-deberta-uncertainty` - With MC Dropout uncertainty (Update 2)