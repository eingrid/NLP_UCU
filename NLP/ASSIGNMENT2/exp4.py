# ============================================================
#  Emotion Classification (Ukrainian) with XLM-RoBERTa
#  + Class Weighting for Imbalanced Data
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np
import torch
import re

from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch import nn

# ============================================================
# 0. Device Configuration
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# 1. Load Dataset
# ============================================================

train_df, val_df = pd.read_csv("data/train_balanced_augmented.csv"), pd.read_csv("data/val.csv")

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("Original training set distribution:")
print(train_df["emotion"].value_counts())
print()

# ============================================================
# 2. Balance training set (oversample minority)
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
print()

# ============================================================
# 3. Compute Class Weights
# ============================================================

# Create label mappings
label2id = {label: i for i, label in enumerate(sorted(train_df["emotion"].unique()))}
id2label = {i: label for label, i in label2id.items()}

# Compute class weights based on the ORIGINAL (unbalanced) distribution
# This helps the model focus more on underrepresented classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(sorted(label2id.values())),
    y=train_df['emotion'].map(label2id).values
)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

print("Class weights:")
for label, weight in zip(sorted(label2id.keys()), class_weights):
    print(f"  {label}: {weight:.3f}")
print()

# ============================================================
# 4. Focal Loss for Imbalanced Data
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss focuses training on hard examples and down-weights easy ones.
    Great for imbalanced datasets where some classes are harder to learn.
    
    Args:
        alpha: Weighting factor (can be class weights or scalar)
        gamma: Focusing parameter. Higher gamma = more focus on hard examples
               gamma=0 reduces to standard cross-entropy
               gamma=2 is typical
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none', weight=self.alpha)(inputs, targets)
        pt = torch.exp(-ce_loss)  # Probability of true class
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FocalLossTrainer(Trainer):
    def __init__(self, *args, alpha=None, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Use Focal Loss
        loss = self.focal_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# ============================================================
# 5. HuggingFace Dataset + Tokenizer
# ============================================================

MODEL = "xlm-roberta-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

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
# 6. Load Model
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
# 7. Training Args with Better Hyperparameters
# ============================================================

training_args = TrainingArguments(
    output_dir="./emotion-roberta-focal",
    num_train_epochs=15,  # More epochs with early stopping
    per_device_train_batch_size=8,  # Smaller batches for better gradients
    gradient_accumulation_steps=2,  # Effective batch size = 16
    per_device_eval_batch_size=32,
    warmup_ratio=0.15,  # More warmup for stability
    learning_rate=2e-5,  # Slightly higher learning rate
    weight_decay=0.01,  # Less regularization
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,  # Only keep best 3 checkpoints
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    fp16=True,  # Mixed precision for faster training
    lr_scheduler_type="cosine",  # Cosine learning rate decay
)

# Use FocalLossTrainer with both focal loss AND class weights
trainer = FocalLossTrainer(
    model=model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    alpha=class_weights_tensor,  # Combine focal loss with class weights
    gamma=2.0,  # Standard focal loss gamma
)

# ============================================================
# 8. Train
# ============================================================

print("Starting training with Focal Loss (gamma=2.0) + class weights...")
print("Hyperparameters:")
print(f"  - Epochs: 15")
print(f"  - Learning rate: 2e-5 with cosine schedule")
print(f"  - Batch size: 8 (with gradient accumulation = effective 16)")
print(f"  - Focal loss gamma: 2.0")
print()

trainer.train()

# ============================================================
# 9. Evaluate
# ============================================================

results = trainer.evaluate()
print("\nEvaluation:", results)

# ============================================================
# 10. Classification Report (detailed)
# ============================================================

raw_preds = trainer.predict(hf_val).predictions
final_preds = raw_preds.argmax(axis=1)

# Convert numeric predictions back to string labels
final_preds_labels = [id2label[pred] for pred in final_preds]

print("\nDetailed classification report:")
print(classification_report(val_df["emotion"], final_preds_labels, target_names=sorted(label2id.keys())))

# ============================================================
# 11. Per-Class Performance Analysis
# ============================================================

print("\nPer-class performance with class weights:")
for label in sorted(label2id.keys()):
    mask = val_df["emotion"] == label
    correct = sum((val_df["emotion"][mask] == final_preds_labels[i]) for i, m in enumerate(mask) if m)
    total = sum(mask)
    accuracy = correct / total if total > 0 else 0
    print(f"  {label}: {accuracy:.3f} ({correct}/{total})")

##
# Results tracking:
# Baseline: 58% F1-macro (10 epochs, xlm-roberta-large, 300 samples/class)
# With class weighting: No improvement
# With Focal Loss (gamma=2) + class weights + better hyperparameters: 55%
#
# Key changes in this version:
# 1. Focal Loss with gamma=2.0 (focuses on hard examples)
# 2. Combined with class weights (alpha parameter)
# 3. Increased epochs to 15 with cosine LR schedule
# 4. Smaller batch size (8) with gradient accumulation
# 5. Higher learning rate (2e-5) with more warmup (0.15)
# 6. Less weight decay (0.01)
# 7. FP16 mixed precision training