# ============================================================
#  Emotion Classification (Ukrainian) with XLM-RoBERTa + Focal Loss
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ============================================================
# 0. Device Configuration
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# Focal Loss Implementation
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================
# Custom Trainer with Focal Loss
# ============================================================

class FocalLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        loss = self.focal_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# ============================================================
# 1. Load Dataset
# ============================================================

train_df, val_df = pd.read_csv("data/train_balanced_augmented.csv"), pd.read_csv("data/val.csv")

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# ============================================================
# 2. Optional: Balance training set (oversample minority)
# ============================================================

TARGET_SIZE = 300  # tune depending on GPU and dataset

balanced_parts = []

for label, df_label in train_df.groupby("emotion"):
    if len(df_label) >= TARGET_SIZE:
        df_res = resample(
            df_label,
            replace=False,
            n_samples=TARGET_SIZE,
            random_state=42,
        )
    else:
        df_res = resample(
            df_label,
            replace=True,
            n_samples=TARGET_SIZE,
            random_state=42,
        )
    balanced_parts.append(df_res)

train_df_balanced = pd.concat(balanced_parts).reset_index(drop=True)

print("Balanced counts:")
print(train_df_balanced["emotion"].value_counts())

# ============================================================
# 3. HuggingFace Dataset + Tokenizer
# ============================================================

MODEL = "xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# label mappings
label2id = {label: i for i, label in enumerate(sorted(train_df["emotion"].unique()))}
id2label = {i: label for label, i in label2id.items()}

# Helper: tokenize
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

# Convert to HF Dataset
hf_train = Dataset.from_pandas(train_df_balanced)
hf_val   = Dataset.from_pandas(val_df)

hf_train = hf_train.map(lambda e: {"label": label2id[e["emotion"]]}, remove_columns=["emotion"])
hf_val   = hf_val.map(lambda e: {"label": label2id[e["emotion"]]}, remove_columns=["emotion"])

hf_train = hf_train.map(tokenize, batched=True)
hf_val   = hf_val.map(tokenize, batched=True)

columns_to_remove = [col for col in ["text", "emojis", "category", "text_processed"] if col in hf_train.column_names]
hf_train = hf_train.remove_columns(columns_to_remove)
hf_val   = hf_val.remove_columns(columns_to_remove)

hf_train.set_format("torch")
hf_val.set_format("torch")

# ============================================================
# 4. Load Model
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

# Move model to GPU if available
model.to(device)

# Metrics
metric_f1 = evaluate.load("f1")
metric_accuracy = evaluate.load("accuracy")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "f1_macro": metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        "accuracy": metric_accuracy.compute(predictions=preds, references=labels)["accuracy"],
    }

# ============================================================
# 5. Training Args
# ============================================================

training_args = TrainingArguments(
    output_dir="./emotion-roberta-focal",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_ratio=0.1,
    learning_rate=2e-5,
    weight_decay=0.05,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
)

trainer = FocalLossTrainer(
    model=model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ============================================================
# 6. Train
# ============================================================

trainer.train()

# ============================================================
# 7. Evaluate
# ============================================================

results = trainer.evaluate()
print("Evaluation:", results)

# ============================================================
# 8. Classification Report (detailed)
# ============================================================

raw_preds = trainer.predict(hf_val).predictions
final_preds = raw_preds.argmax(axis=1)

# Convert numeric predictions back to string labels
final_preds_labels = [id2label[pred] for pred in final_preds]

print("\nDetailed classification report:")
print(classification_report(val_df["emotion"], final_preds_labels, target_names=sorted(label2id.keys())))


##
# 10 epochs without preprocess, with augmented data, 300 samples per class, Focal Loss (alpha=1, gamma=2) 0.5 f1