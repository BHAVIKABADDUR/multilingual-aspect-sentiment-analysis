"""
Optimized Sentiment Analysis Model
===================================
This script analyzes customer reviews to determine if they are:
- POSITIVE (customer is happy)
- NEGATIVE (customer is unhappy)  
- NEUTRAL (customer is okay)

Features:
- Smart data balancing
- Advanced text preprocessing
- Optimized TF-IDF features
- Random Forest classifier for better multilingual handling

Author: Bhavika Baddur
Project: E-commerce Multilingual Sentiment Analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import sys

# ============================================================================
# ERROR HANDLING AND VALIDATION
# ============================================================================

def check_file_exists(filepath):
    """Check if required file exists"""
    if not os.path.exists(filepath):
        print(f"\n❌ ERROR: File not found: {filepath}")
        print("\nPlease ensure you have run the data preparation steps:")
        print("  1. python scripts/build_final_dataset.py")
        print("  2. python scripts/05_aspect_extraction.py (creates cleaned dataset)")
        sys.exit(1)

def check_required_columns(df, required_cols):
    """Check if dataframe has required columns"""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\n❌ ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

# ============================================================================
# STEP 1: LOAD THE DATA
# ============================================================================
print("=" * 70)
print("STEP 1: Loading your dataset...")
print("=" * 70)

# Check if file exists
data_file = 'processed_data/enhanced_final_dataset_cleaned.csv'
check_file_exists(data_file)

try:
    # Load the dataset you already prepared
    df = pd.read_csv(data_file, encoding='utf-8')
    print(f"✅ Loaded {len(df):,} reviews successfully!")
    
    # Validate required columns
    required_columns = ['review_text', 'sentiment']
    check_required_columns(df, required_columns)
    
except Exception as e:
    print(f"\n❌ ERROR: Failed to load dataset: {str(e)}")
    sys.exit(1)

# ============================================================================
# STEP 2: ADVANCED TEXT PREPROCESSING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Advanced text preprocessing...")
print("=" * 70)

def clean_text(text):
    """Enhanced text cleaning for better accuracy"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove URLs and emails
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep only letters, numbers, and spaces (preserve Hindi transliterations)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove very short words (likely noise)
    words = text.split()
    words = [w for w in words if len(w) > 2]
    text = ' '.join(words)
    
    return text.strip()

print("🔄 Cleaning review text...")
df['cleaned_text'] = df['review_text'].apply(clean_text)

# Remove very short reviews (less than 3 words)
df = df[df['cleaned_text'].str.split().str.len() >= 3]
print(f"✅ Cleaned dataset: {len(df):,} reviews")

# ============================================================================
# STEP 3: SMART DATA BALANCING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Balancing the dataset...")
print("=" * 70)

print("\n📊 BEFORE Balancing:")
sentiment_counts = df['sentiment'].value_counts()
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {sentiment}: {count:,} ({percentage:.1f}%)")

# Separate by sentiment
df_positive = df[df['sentiment'] == 'positive']
df_negative = df[df['sentiment'] == 'negative']
df_neutral = df[df['sentiment'] == 'neutral']

# Smart balancing: Make minority classes = 70% of majority
target_size = int(len(df_positive) * 0.7)

print("\n🔄 Applying smart balancing (70% of majority class)...")

df_negative_balanced = resample(df_negative, 
                                replace=True,
                                n_samples=min(target_size, len(df_negative) * 2),
                                random_state=42)

df_neutral_balanced = resample(df_neutral,
                              replace=True,
                              n_samples=min(target_size, len(df_neutral) * 3),
                              random_state=42)

# Combine balanced dataset
df_balanced = pd.concat([df_positive, df_negative_balanced, df_neutral_balanced])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n📊 AFTER Balancing:")
balanced_counts = df_balanced['sentiment'].value_counts()
for sentiment, count in balanced_counts.items():
    percentage = (count / len(df_balanced)) * 100
    print(f"  {sentiment}: {count:,} ({percentage:.1f}%)")

print(f"\n✅ Balanced dataset size: {len(df_balanced):,} reviews")

# Extract the columns we need
X = df_balanced['cleaned_text']  # Reviews (input)
y = df_balanced['sentiment']     # Sentiment labels (output)

# ============================================================================
# STEP 4: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Splitting data into training and testing sets...")
print("=" * 70)

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducible results
    stratify=y          # Keep same proportion of sentiments in both sets
)

print(f"✅ Training set: {len(X_train):,} reviews")
print(f"✅ Testing set: {len(X_test):,} reviews")

# ============================================================================
# STEP 5: OPTIMIZED TF-IDF FEATURE EXTRACTION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Creating optimized features (TF-IDF)...")
print("=" * 70)

# Optimized TF-IDF with better parameters
vectorizer = TfidfVectorizer(
    max_features=8000,      # ⬆️ More features for better accuracy
    ngram_range=(1, 2),     # Use single words and word pairs
    min_df=3,               # ⬆️ Filter rare words better
    max_df=0.75,            # ⬆️ Filter common words better
    sublinear_tf=True,      # ✨ Use logarithmic term frequency
    use_idf=True,           # Use inverse document frequency
    norm='l2'               # L2 normalization
)

print("🔄 Processing reviews...")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"✅ Created {X_train_tfidf.shape[1]:,} features from {len(X_train):,} reviews")

# ============================================================================
# STEP 6: TRAIN OPTIMIZED MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Training optimized sentiment analysis model...")
print("=" * 70)
print("🧠 Using Random Forest for better multilingual handling...")
print("   (This will take 2-3 minutes)")

# Random Forest works better with code-mixed text
model = RandomForestClassifier(
    n_estimators=100,         # ✨ 100 decision trees
    max_depth=50,             # Maximum tree depth
    min_samples_split=5,      # Minimum samples to split
    min_samples_leaf=2,       # Minimum samples in leaf
    random_state=42,
    class_weight='balanced_subsample',  # Handle any remaining imbalance
    n_jobs=-1,                # Use all CPU cores
    verbose=1
)

# Train the model
model.fit(X_train_tfidf, y_train)
print("\n✅ Model training complete!")

# ============================================================================
# STEP 7: TEST THE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: Testing the model on unseen reviews...")
print("=" * 70)

# Predict sentiments for test set
y_pred = model.predict(X_test_tfidf)
print(f"✅ Made predictions for {len(y_test):,} reviews")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 MODEL ACCURACY: {accuracy * 100:.2f}%")
print(f"   This means the model correctly identified sentiment in {accuracy * 100:.1f}% of reviews!")

# ============================================================================
# STEP 8: DETAILED RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: Detailed Performance Report")
print("=" * 70)

# Classification report
print("\n📊 Performance by Sentiment Type:")
# Get unique labels from the data
unique_labels = sorted(y_test.unique())
print(classification_report(y_test, y_pred, labels=unique_labels))

print("\nWhat do these metrics mean?")
print("  • Precision: When model says 'positive', how often is it right?")
print("  • Recall: Of all actual 'positive' reviews, how many did model find?")
print("  • F1-Score: Overall performance (balance of precision and recall)")

# ============================================================================
# STEP 9: CONFUSION MATRIX (VISUAL)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: Creating confusion matrix visualization...")
print("=" * 70)

# Calculate confusion matrix
# Get unique labels from the data
unique_labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

# Create visualization
plt.figure(figsize=(10, 8))
label_names = [str(label).title() for label in unique_labels]
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=label_names,
            yticklabels=label_names,
            cbar_kws={'label': 'Number of Reviews'})
plt.title('Confusion Matrix: Sentiment Analysis Results\nby Bhavika Baddur', 
          fontsize=14, weight='bold', pad=20)
plt.ylabel('Actual Sentiment', fontsize=12)
plt.xlabel('Predicted Sentiment', fontsize=12)

# Save the plot
os.makedirs('reports/images', exist_ok=True)
plt.savefig('reports/images/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved: reports/images/confusion_matrix.png")
plt.close()

print("\n📊 How to read the confusion matrix:")
print("   • Diagonal (dark blue) = Correct predictions")
print("   • Off-diagonal = Mistakes (e.g., predicted positive but actually negative)")

# ============================================================================
# STEP 10: TEST WITH NEW REVIEWS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 10: Testing with sample reviews...")
print("=" * 70)

# Test with some example reviews
test_reviews = [
    "Product quality amazing hai! Fast delivery bhi. Very happy customer.",
    "Worst product ever. Totally disappointed. Waste of money.",
    "Product okay hai. Nothing special. Average quality.",
    "Delivery bahut late tha but product achha hai.",
    "Excellent! Worth every penny. Highly recommend karta hun.",
    "Terrible service. Product kharab tha. Not satisfied at all.",
    "Decent product. Price is reasonable. Can buy again."
]

print("\n🔍 Predicting sentiment for new reviews:\n")

for i, review in enumerate(test_reviews, 1):
    # Clean and convert review to TF-IDF
    cleaned = clean_text(review)
    review_tfidf = vectorizer.transform([cleaned])
    
    # Predict sentiment
    prediction = model.predict(review_tfidf)[0]
    
    # Get confidence scores
    probabilities = model.predict_proba(review_tfidf)[0]
    confidence = max(probabilities) * 100
    
    print(f"Review {i}: {review}")
    print(f"  → Predicted: {prediction.upper()} (Confidence: {confidence:.1f}%)")
    print()

# ============================================================================
# STEP 11: SAVE THE MODEL
# ============================================================================
print("=" * 70)
print("STEP 11: Saving the model for future use...")
print("=" * 70)

import pickle

# Create models directory
os.makedirs('models', exist_ok=True)

# Save the model
with open('models/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved: models/sentiment_model.pkl")

# Save the vectorizer
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✅ Vectorizer saved: models/tfidf_vectorizer.pkl")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("🎉 ANALYSIS COMPLETE!")
print("=" * 70)

print(f"\n✅ MODEL SUMMARY:")
print(f"   • Model Type: Random Forest (100 trees)")
print(f"   • Trained on: {len(X_train):,} reviews (balanced)")
print(f"   • Tested on: {len(X_test):,} reviews")
print(f"   • Accuracy: {accuracy * 100:.2f}%")
print(f"   • Features: {X_train_tfidf.shape[1]:,}")

print(f"\n📁 FILES CREATED:")
print(f"   • models/sentiment_model.pkl (trained model)")
print(f"   • models/tfidf_vectorizer.pkl (text processor)")
print(f"   • reports/images/confusion_matrix.png (visualization)")

print(f"\n🎓 IMPROVEMENTS APPLIED:")
print(f"   ✅ Smart data balancing (70% balancing strategy)")
print(f"   ✅ Advanced text preprocessing (noise removal)")
print(f"   ✅ Optimized TF-IDF (8000 features, bigrams)")
print(f"   ✅ Random Forest classifier (better for multilingual)")

print(f"\n🚀 NEXT STEPS:")
print(f"   1. Check the confusion matrix image")
print(f"   2. Ready for Option 2: Aspect Extraction")
print(f"   3. Or Option 3: Build Interactive Dashboard")
print(f"   4. Or Option 4: Advanced models (IndicBERT) for 80%+ accuracy")

print("\n" + "=" * 70)
print("Project by: Bhavika Baddur")
print("Multilingual E-commerce Sentiment Analysis Platform")
print("=" * 70)
