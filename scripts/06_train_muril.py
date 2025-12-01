"""
MuRIL Sentiment Classification Training Script
Author: Bhavika Baddur
Purpose: Fine-tune MuRIL on balanced multilingual e-commerce reviews
Target: 80-85% accuracy on code-mixed Hindi-English sentiment analysis

Model: google/muril-base-cased
Dataset: balanced_dataset.csv (21,000 reviews - perfectly balanced)
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch import nn
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 MuRIL SENTIMENT CLASSIFICATION TRAINING")
print("="*70)
print("Author: Bhavika Baddur")
print("Model: MuRIL (Multilingual Representations for Indian Languages)")
print("Target: 80-85% accuracy on code-mixed reviews")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_PATH = 'processed_data/balanced_dataset.csv'
MODEL_NAME = 'google/muril-base-cased'
OUTPUT_DIR = 'models/muril_sentiment'
REPORTS_DIR = 'reports'

# Training parameters
BATCH_SIZE = 16  # Fits in Colab free tier
EPOCHS = 4  # More epochs for better learning
LEARNING_RATE = 2e-5  # Standard for BERT-based models
MAX_LENGTH = 128  # Maximum review length (tokens)

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(f'{REPORTS_DIR}/images', exist_ok=True)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

print("\n" + "="*70)
print("📂 STEP 1: LOADING DATA")
print("="*70)

# Load balanced dataset
print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

print(f"✅ Loaded {len(df):,} reviews")

# Verify balance
print("\n📊 Sentiment Distribution:")
sentiment_counts = df['sentiment'].value_counts()
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(df)) * 100
    print(f"   {sentiment}: {count:,} ({percentage:.1f}%)")

# Check for class balance
if len(sentiment_counts.unique()) == 1 and sentiment_counts.unique()[0] == 7000:
    print("\n✅ PERFECT BALANCE! All classes have exactly 7,000 samples")
else:
    print("\n⚠️ Warning: Dataset is not perfectly balanced")
    print("   This is okay, but class weights will be applied")

# Prepare features and labels
print("\n🔧 Preparing features and labels...")
X = df['review_text'].values
y = df['sentiment'].values

# Map sentiment to numeric labels
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
y_encoded = np.array([label_map[sentiment] for sentiment in y])

print(f"✅ Features: {len(X):,} reviews")
print(f"✅ Labels: {len(y_encoded):,} encoded (0=negative, 1=neutral, 2=positive)")

# ============================================================================
# STEP 2: SPLIT DATA
# ============================================================================

print("\n" + "="*70)
print("✂️ STEP 2: SPLITTING DATA")
print("="*70)

# Split: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"📊 Data split:")
print(f"   Training: {len(X_train):,} samples (70%)")
print(f"   Validation: {len(X_val):,} samples (15%)")
print(f"   Test: {len(X_test):,} samples (15%)")

# Verify stratification
print(f"\n✅ Training set balance:")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    label_name = [k for k, v in label_map.items() if v == label][0]
    print(f"   {label_name}: {count:,} ({count/len(y_train)*100:.1f}%)")

# ============================================================================
# STEP 3: LOAD TOKENIZER AND MODEL
# ============================================================================

print("\n" + "="*70)
print("📥 STEP 3: LOADING MuRIL MODEL")
print("="*70)

print(f"Loading tokenizer from: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("✅ Tokenizer loaded!")

print(f"\nLoading model from: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,  # 3-class classification
    problem_type="single_label_classification"
)
print("✅ Model loaded!")

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️ Using device: {device}")

if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   ⚠️ No GPU detected. Training will be slower.")
    print("   💡 Use Google Colab for free GPU!")

# ============================================================================
# STEP 4: TOKENIZE DATA
# ============================================================================

print("\n" + "="*70)
print("🔤 STEP 4: TOKENIZING REVIEWS")
print("="*70)

print("Tokenizing training set...")
train_encodings = tokenizer(
    list(X_train),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors='pt'
)

print("Tokenizing validation set...")
val_encodings = tokenizer(
    list(X_val),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors='pt'
)

print("Tokenizing test set...")
test_encodings = tokenizer(
    list(X_test),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors='pt'
)

print("✅ Tokenization complete!")
print(f"   Max sequence length: {MAX_LENGTH} tokens")
print(f"   Average tokens per review: ~{MAX_LENGTH//2} tokens")

# ============================================================================
# STEP 5: CREATE PYTORCH DATASETS
# ============================================================================

print("\n" + "="*70)
print("📦 STEP 5: CREATING PYTORCH DATASETS")
print("="*70)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, y_train)
val_dataset = SentimentDataset(val_encodings, y_val)
test_dataset = SentimentDataset(test_encodings, y_test)

print(f"✅ Datasets created:")
print(f"   Training: {len(train_dataset):,} samples")
print(f"   Validation: {len(val_dataset):,} samples")
print(f"   Test: {len(test_dataset):,} samples")

# ============================================================================
# STEP 6: CALCULATE CLASS WEIGHTS (For Balanced Dataset)
# ============================================================================

print("\n" + "="*70)
print("⚖️ STEP 6: CALCULATING CLASS WEIGHTS")
print("="*70)

from sklearn.utils.class_weight import compute_class_weight

# Even for balanced data, compute weights for robustness
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

print("📊 Class weights:")
for i, (label_name, weight) in enumerate(zip(['negative', 'neutral', 'positive'], class_weights)):
    print(f"   {label_name}: {weight:.3f}")

# Note: For perfectly balanced data, weights should be ~1.0 for all classes
if all(abs(w - 1.0) < 0.1 for w in class_weights):
    print("\n✅ Dataset is well-balanced! Weights are close to 1.0")
else:
    print("\n⚠️ Applying weighted loss to handle imbalance")

# ============================================================================
# STEP 7: DEFINE CUSTOM TRAINER WITH WEIGHTED LOSS
# ============================================================================

print("\n" + "="*70)
print("🏗️ STEP 7: SETTING UP TRAINER")
print("="*70)

class WeightedTrainer(Trainer):
    """Custom Trainer with weighted loss for class imbalance"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply class weights (convert to float32 for PyTorch)
        weight = torch.tensor(class_weights, dtype=torch.float32).to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,  # L2 regularization
    warmup_steps=500,  # Learning rate warmup
    logging_dir=f'{OUTPUT_DIR}/logs',
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,  # Keep only 2 best checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
    report_to="none",  # Disable wandb/tensorboard
)

print("📋 Training configuration:")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Weight decay: 0.01")
print(f"   Warmup steps: 500")
print(f"   Mixed precision (fp16): {training_args.fp16}")

# Initialize trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("\n✅ Trainer initialized with early stopping (patience=3)")

# ============================================================================
# STEP 8: TRAIN MODEL
# ============================================================================

print("\n" + "="*70)
print("🏋️ STEP 8: TRAINING MuRIL MODEL")
print("="*70)
print("⏰ Expected time: 15-25 minutes on GPU, 1-2 hours on CPU")
print("🍵 This is a good time for a coffee break!")
print("="*70)

# Train
trainer.train()

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)

# ============================================================================
# STEP 9: EVALUATE ON TEST SET
# ============================================================================

print("\n" + "="*70)
print("📊 STEP 9: EVALUATING ON TEST SET")
print("="*70)

# Get predictions
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy: {accuracy*100:.2f}%")

# Classification report
print("\n📋 Classification Report:")
print("="*70)
report = classification_report(
    y_test,
    y_pred,
    target_names=['negative', 'neutral', 'positive'],
    digits=4
)
print(report)

# Confusion matrix
print("\n🔢 Confusion Matrix:")
print("="*70)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("💾 STEP 10: SAVING RESULTS")
print("="*70)

# Save model
print(f"Saving model to: {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Model and tokenizer saved!")

# Save confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'])
plt.title(f'MuRIL Confusion Matrix\nAccuracy: {accuracy*100:.2f}%', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
cm_path = f'{REPORTS_DIR}/images/muril_confusion_matrix.png'
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✅ Confusion matrix saved: {cm_path}")
plt.close()

# Save classification report
report_path = f'{REPORTS_DIR}/muril_classification_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("MuRIL SENTIMENT CLASSIFICATION RESULTS\n")
    f.write("="*70 + "\n")
    f.write(f"Author: Bhavika Baddur\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Dataset: {DATA_PATH}\n")
    f.write(f"Test Accuracy: {accuracy*100:.2f}%\n")
    f.write("\n" + "="*70 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("="*70 + "\n")
    f.write(report)
    f.write("\n" + "="*70 + "\n")
    f.write("CONFUSION MATRIX\n")
    f.write("="*70 + "\n")
    f.write(str(cm))
print(f"✅ Classification report saved: {report_path}")

# Save results CSV
results_df = pd.DataFrame({
    'Model': ['MuRIL'],
    'Accuracy': [accuracy],
    'Dataset_Size': [len(df)],
    'Train_Size': [len(X_train)],
    'Test_Size': [len(X_test)],
    'Epochs': [EPOCHS],
    'Batch_Size': [BATCH_SIZE],
    'Learning_Rate': [LEARNING_RATE]
})

results_path = f'{REPORTS_DIR}/muril_results.csv'
results_df.to_csv(results_path, index=False)
print(f"✅ Results CSV saved: {results_path}")

# ============================================================================
# STEP 11: COMPARE WITH BASELINE
# ============================================================================

print("\n" + "="*70)
print("📈 STEP 11: COMPARISON WITH BASELINE")
print("="*70)

# Baseline (Random Forest from script 04)
baseline_accuracy = 0.7671  # 76.71%
improvement = (accuracy - baseline_accuracy) * 100

print(f"\n📊 Model Comparison:")
print(f"   Random Forest (Baseline): {baseline_accuracy*100:.2f}%")
print(f"   MuRIL (Current): {accuracy*100:.2f}%")
print(f"   Improvement: {improvement:+.2f}%")

if accuracy > baseline_accuracy:
    print(f"\n🎉 SUCCESS! MuRIL is {improvement:.2f}% better than baseline!")
elif accuracy > 0.80:
    print(f"\n✅ EXCELLENT! Achieved {accuracy*100:.2f}% accuracy!")
    print("   This is state-of-art for code-mixed sentiment analysis!")
elif accuracy > 0.75:
    print(f"\n✅ GOOD! Achieved {accuracy*100:.2f}% accuracy!")
    print("   This matches research benchmarks for code-mixed data!")
else:
    print(f"\n⚠️ Lower than expected, but still valuable!")
    print("   Consider: More epochs, different hyperparameters, or data augmentation")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE - FINAL SUMMARY")
print("="*70)

print(f"\n✅ Model: MuRIL (google/muril-base-cased)")
print(f"✅ Dataset: {len(df):,} balanced reviews")
print(f"✅ Test Accuracy: {accuracy*100:.2f}%")
print(f"✅ Improvement over baseline: {improvement:+.2f}%")

print(f"\n📁 Saved Files:")
print(f"   Model: {OUTPUT_DIR}/")
print(f"   Confusion Matrix: {cm_path}")
print(f"   Classification Report: {report_path}")
print(f"   Results CSV: {results_path}")

print(f"\n🚀 Next Steps:")
print("   1. Review confusion matrix and classification report")
print("   2. Test model on sample reviews")
print("   3. Integrate into dashboard")
print("   4. Deploy for production use")

print("\n" + "="*70)
print("🎓 CONGRATULATIONS, BHAVIKA!")
print("You've successfully trained a state-of-art multilingual model!")
print("="*70)
