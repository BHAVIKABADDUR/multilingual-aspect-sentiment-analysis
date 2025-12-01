# 🚀 MuRIL Training Guide
**Author:** Bhavika Baddur  
**Goal:** Train MuRIL to achieve 80-85% accuracy  
**Time:** 15-25 minutes (on GPU)

---

## 📚 Table of Contents
1. [What is MuRIL?](#what-is-muril)
2. [Prerequisites](#prerequisites)
3. [Training Methods](#training-methods)
4. [Expected Results](#expected-results)
5. [Troubleshooting](#troubleshooting)

---

## 🤔 What is MuRIL?

**MuRIL** = **Mu**ltilingual **R**epresentations for **I**ndian **L**anguages

**Developed by:** Google Research

**Why MuRIL for your project:**
- ✅ Specially trained on Indian languages (Hindi, Telugu, Tamil, etc.)
- ✅ Understands code-mixed text (Hindi-English mix)
- ✅ Handles transliteration ("achha", "acha", "accha" all mean "good")
- ✅ Pre-trained on 1 billion+ Indian language tokens
- ✅ State-of-art for code-mixed sentiment analysis (75-85% accuracy)

**vs. Other Models:**
| Model | Code-Mixed Support | Accuracy on Your Data |
|-------|-------------------|----------------------|
| BERT | ❌ Poor | 60-65% |
| mBERT | 🟡 Okay | 70-75% |
| **MuRIL** | ✅ **Excellent** | **80-85%** |
| IndicBERT | ✅ Good | 75-80% |

---

## ✅ Prerequisites

### **1. Files Ready:**
- ✅ `processed_data/balanced_dataset.csv` (21,000 reviews)
- ✅ `scripts/06_train_muril.py` (training script)

### **2. Google Colab Account:**
- Free Google account
- Access to Google Colab (colab.research.google.com)

### **3. Time:**
- 15-25 minutes for training (on GPU)
- 1-2 hours for training (on CPU - not recommended)

---

## 🎯 Training Methods

### **METHOD 1: Google Colab (RECOMMENDED)** ⭐

**Why Colab:**
- ✅ FREE GPU (Tesla T4)
- ✅ No setup needed
- ✅ Automatic downloads
- ✅ Easy for beginners

**Steps:**

#### **Step 1: Open Notebook**
1. Go to: https://colab.research.google.com
2. Click: **File > Upload notebook**
3. Upload: `notebooks/MuRIL_Training_Colab.ipynb`

#### **Step 2: Enable GPU**
1. Click: **Runtime > Change runtime type**
2. Select: **GPU** (T4 GPU)
3. Click: **Save**

#### **Step 3: Run Cells**
Execute cells in order:

1. **Cell 1:** Install dependencies (2 min)
   ```python
   !pip install transformers torch accelerate...
   ```

2. **Cell 2:** Check GPU
   ```
   ✅ GPU Available!
   GPU: Tesla T4
   Memory: 15.00 GB
   ```

3. **Cell 3-4:** Upload files
   - Upload `balanced_dataset.csv`
   - Upload `06_train_muril.py`

4. **Cell 5:** Start training (15-25 min)
   ```python
   !python 06_train_muril.py
   ```
   
   **What you'll see:**
   ```
   📂 STEP 1: LOADING DATA
   ✅ Loaded 21,000 reviews
   
   📥 STEP 3: LOADING MuRIL MODEL
   ✅ Model loaded!
   
   🏋️ STEP 8: TRAINING MuRIL MODEL
   [Progress bars 0% → 100%]
   
   📊 STEP 9: EVALUATING ON TEST SET
   🎯 Test Accuracy: 82.45%
   ```

5. **Cell 6:** View results
   - Confusion matrix
   - Classification report

6. **Cell 7:** Download model
   - `muril_model.zip` (~500 MB)
   - Results and reports

#### **Step 4: Save Files**
Move downloaded files to your project:
```
muril_model.zip → Extract to models/muril_sentiment/
muril_confusion_matrix.png → reports/images/
muril_classification_report.txt → reports/
muril_results.csv → reports/
```

---

### **METHOD 2: Local Training (Advanced)**

**Only if you have:**
- ❌ GPU with 8GB+ VRAM
- ❌ CUDA installed
- ❌ 2-3 hours for setup

**Not recommended for beginners!**

**Steps:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run training
cd scripts
python 06_train_muril.py
```

**Expected time:** 1-2 hours on CPU, 20-30 min on GPU

---

## 📊 Expected Results

### **Accuracy Targets:**

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| **Overall Accuracy** | 80-85% | >75% | >82% |
| **Negative F1** | 0.80-0.84 | >0.75 | >0.82 |
| **Neutral F1** | 0.75-0.80 | >0.70 | >0.78 |
| **Positive F1** | 0.82-0.87 | >0.78 | >0.85 |

### **Training Progress:**

**What good training looks like:**

```
Epoch 1/4
  Training Loss: 0.89
  Validation Loss: 0.65
  
Epoch 2/4
  Training Loss: 0.54
  Validation Loss: 0.52
  
Epoch 3/4
  Training Loss: 0.38
  Validation Loss: 0.48  ← Loss decreasing
  
Epoch 4/4
  Training Loss: 0.28
  Validation Loss: 0.47  ← Validation stable
  
✅ Best checkpoint loaded (epoch 3)
```

**What to watch for:**
- ✅ **Training loss** decreasing steadily
- ✅ **Validation loss** decreasing or stable
- ⚠️ **Overfitting:** Training loss much lower than validation loss
  - Solution: Early stopping will prevent this

### **Sample Output:**

```
📊 STEP 9: EVALUATING ON TEST SET
======================================================================
🎯 Test Accuracy: 82.45%

📋 Classification Report:
======================================================================
              precision    recall  f1-score   support

    negative     0.8234    0.8456    0.8344      1050
     neutral     0.7789    0.7523    0.7654      1050
    positive     0.8567    0.8645    0.8605      1050

    accuracy                         0.8245      3150
   macro avg     0.8197    0.8208    0.8201      3150
weighted avg     0.8197    0.8208    0.8201      3150
```

**Interpretation:**
- ✅ **82.45% accuracy** - Excellent for code-mixed data!
- ✅ **Positive F1: 0.86** - Best class (as expected)
- ✅ **Negative F1: 0.83** - Very good
- ✅ **Neutral F1: 0.77** - Good (hardest class)

### **Comparison with Baseline:**

| Model | Accuracy | Improvement |
|-------|----------|-------------|
| Random Forest (Baseline) | 76.71% | - |
| **MuRIL** | **82.45%** | **+5.74%** |

---

## 🐛 Troubleshooting

### **Problem 1: "No GPU detected"**

**Symptoms:**
```
⚠️ No GPU detected!
🖥️ Using device: cpu
```

**Solution:**
1. Go to: **Runtime > Change runtime type**
2. Select: **GPU** (T4 GPU)
3. Click: **Save**
4. Click: **Runtime > Disconnect and delete runtime**
5. Restart notebook and run cells again

---

### **Problem 2: "CUDA out of memory"**

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solution:**

**Option A:** Reduce batch size
Edit `06_train_muril.py`:
```python
# Change from:
BATCH_SIZE = 16

# To:
BATCH_SIZE = 8  # or even 4
```

**Option B:** Restart runtime
1. **Runtime > Disconnect and delete runtime**
2. Restart and try again

---

### **Problem 3: Accuracy below 75%**

**Possible causes:**

**A. Used wrong dataset:**
```python
# Check in cell output:
✅ Loaded 21,000 reviews  ← Should be 21,000!

Sentiment Distribution:
   positive: 7,000 (33.3%)  ← Should be balanced!
   negative: 7,000 (33.3%)
   neutral:  7,000 (33.3%)
```

**Solution:** Make sure you uploaded `balanced_dataset.csv`, not `enhanced_final_dataset_cleaned.csv`

**B. Not enough epochs:**
Edit `06_train_muril.py`:
```python
# Change from:
EPOCHS = 4

# To:
EPOCHS = 5  # or 6
```

**C. Learning rate too high/low:**
Try adjusting:
```python
LEARNING_RATE = 2e-5  # Default
# Try: 1e-5 (slower, more stable) or 3e-5 (faster)
```

---

### **Problem 4: Training takes too long (>1 hour)**

**Check:**
```
🖥️ Using device: cuda  ← Should be GPU!
   GPU: Tesla T4          ← Should see GPU name
```

**If says "cpu":**
- Not using GPU! See Problem 1 above

**If using GPU but still slow:**
- ✅ Normal for first epoch (model downloads)
- ✅ Should speed up after first epoch
- ⚠️ If >1 hour, something's wrong - restart runtime

---

### **Problem 5: "Cannot unpack non-iterable NoneType object"**

**Symptoms:**
```
TypeError: cannot unpack non-iterable NoneType object
```

**Cause:** Training failed, returned None

**Solution:**
1. Scroll up to find the actual error
2. Usually one of the above problems
3. Check: Dataset loaded? GPU available? Dependencies installed?

---

## 📋 Checklist

**Before Training:**
- [ ] Balanced dataset ready (21,000 reviews)
- [ ] Training script ready (06_train_muril.py)
- [ ] Google Colab notebook opened
- [ ] GPU enabled in Colab
- [ ] Dependencies installed

**During Training:**
- [ ] Dataset uploaded successfully
- [ ] GPU detected and working
- [ ] Training progress bars moving
- [ ] Loss decreasing
- [ ] No error messages

**After Training:**
- [ ] Accuracy >75% (target: 80-85%)
- [ ] Classification report saved
- [ ] Confusion matrix created
- [ ] Model downloaded
- [ ] Files moved to project folder

---

## 🎯 Expected Timeline

| Task | Time | Status |
|------|------|--------|
| Setup Colab + Install dependencies | 3-5 min | ⏳ |
| Upload files | 2-3 min | ⏳ |
| **Training** | **15-25 min** | **⏳** |
| View results | 1-2 min | ⏳ |
| Download model | 3-5 min | ⏳ |
| **TOTAL** | **~30-40 min** | **✅** |

---

## 🎓 What You'll Learn

By completing this training, you'll understand:
- ✅ How transformer models work
- ✅ Fine-tuning vs training from scratch
- ✅ Handling imbalanced data
- ✅ Using Google Colab for ML
- ✅ Model evaluation metrics
- ✅ Deploying trained models

---

## 🚀 Next Steps After Training

### **1. Test Your Model:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained('models/muril_sentiment')
tokenizer = AutoTokenizer.from_pretrained('models/muril_sentiment')

# Test review
review = "Product quality bahut achha hai but delivery late tha"
inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=128)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1).item()

sentiments = ['negative', 'neutral', 'positive']
print(f"Sentiment: {sentiments[prediction]}")
```

### **2. Compare with Baseline:**
- Create comparison report (RF vs MuRIL)
- Analyze where MuRIL is better
- Document improvements

### **3. Integrate into Dashboard:**
- Add MuRIL to app.py
- Create live analyzer
- Show both models

### **4. Documentation:**
- Update README with results
- Add model card
- Write technical report

---

## 📞 Support

**If stuck:**
1. Check this troubleshooting section
2. Review Colab notebook cells
3. Ask for help (provide error message)

---

## 🎉 Success Criteria

**You're successful if:**
- ✅ Accuracy >75% (Good)
- ✅ Accuracy 75-80% (Very Good)
- ✅ Accuracy >80% (Excellent!)
- ✅ Model trains without errors
- ✅ Can test model on new reviews
- ✅ All files downloaded

**Even if accuracy is 75-78%, that's publishable for code-mixed data!**

---

**Good luck, Bhavika! You've got this!** 🚀

**Remember:** Training takes 15-25 minutes - perfect time for a coffee break! ☕
