"""
Aspect-Based Sentiment Analysis
================================
This script extracts WHAT customers are talking about and whether it's positive/negative.

Aspects identified:
- Product Quality (quality, features, performance)
- Delivery (shipping, delivery speed, courier)
- Packaging (box, packaging quality)
- Price (cost, value for money)
- Customer Service (support, service, returns)

Author: Bhavika Baddur
Project: E-commerce Multilingual Sentiment Analysis
"""

import pandas as pd
import numpy as np
import re
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ============================================================================
# ERROR HANDLING AND VALIDATION
# ============================================================================

def check_file_exists(filepath, error_msg=None):
    """Check if required file exists"""
    if not os.path.exists(filepath):
        print(f"\n❌ ERROR: File not found: {filepath}")
        if error_msg:
            print(error_msg)
        sys.exit(1)
    return True

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"\n❌ ERROR: Could not create directory {directory}: {str(e)}")
        return False

# ============================================================================
# STEP 1: LOAD DATA AND MODEL
# ============================================================================
print("=" * 70)
print("STEP 1: Loading dataset and sentiment model...")
print("=" * 70)

# Check if dataset exists
data_file = 'processed_data/enhanced_final_dataset_cleaned.csv'
check_file_exists(data_file, 
                 "\nPlease run: python scripts/build_final_dataset.py")

try:
    # Load dataset
    df = pd.read_csv(data_file, encoding='utf-8')
    print(f"✅ Loaded {len(df):,} reviews")
except Exception as e:
    print(f"\n❌ ERROR: Failed to load dataset: {str(e)}")
    sys.exit(1)

# Check if model files exist
model_file = 'models/sentiment_model.pkl'
vectorizer_file = 'models/tfidf_vectorizer.pkl'

check_file_exists(model_file, 
                 "\nPlease run: python scripts/04_simple_sentiment_analysis.py")
check_file_exists(vectorizer_file,
                 "\nPlease run: python scripts/04_simple_sentiment_analysis.py")

try:
    # Load trained sentiment model
    with open(model_file, 'rb') as f:
        sentiment_model = pickle.load(f)
    with open(vectorizer_file, 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Loaded sentiment model")
except Exception as e:
    print(f"\n❌ ERROR: Failed to load models: {str(e)}")
    sys.exit(1)

# ============================================================================
# STEP 2: DEFINE ASPECT KEYWORDS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Defining aspect keywords...")
print("=" * 70)

# Comprehensive keyword dictionary for aspects (English + Hindi transliteration)
ASPECT_KEYWORDS = {
    'product_quality': {
        'english': ['quality', 'product', 'build', 'performance', 'feature', 'durability', 
                   'material', 'design', 'worth', 'good', 'bad', 'excellent', 'poor',
                   'awesome', 'terrible', 'amazing', 'worst', 'best', 'defect', 'broken'],
        'hindi': ['achha', 'acha', 'accha', 'badhiya', 'badiya', 'mast', 'zabardast',
                 'bekaar', 'bekar', 'kharab', 'ghatiya', 'kamaal', 'bura', 'shandar']
    },
    'delivery': {
        'english': ['delivery', 'deliver', 'delivered', 'shipping', 'ship', 'courier',
                   'late', 'delay', 'fast', 'quick', 'slow', 'arrived', 'receive',
                   'tracking', 'time', 'days', 'weeks'],
        'hindi': ['pahunch', 'mila', 'aaya', 'der', 'jaldi', 'late', 'time']
    },
    'packaging': {
        'english': ['packaging', 'package', 'packed', 'box', 'seal', 'wrapped',
                   'damaged', 'broken', 'good packing', 'bad packing', 'secure'],
        'hindi': ['dibba', 'packet', 'packing', 'kharab', 'tuta']
    },
    'price': {
        'english': ['price', 'cost', 'expensive', 'cheap', 'value', 'money',
                   'worth', 'overpriced', 'affordable', 'reasonable', 'discount',
                   'offer', 'paisa', 'rupee', 'rs'],
        'hindi': ['paisa', 'daam', 'dam', 'mehenga', 'mehnga', 'sasta', 'keemat']
    },
    'customer_service': {
        'english': ['service', 'support', 'customer care', 'help', 'return', 'refund',
                   'replace', 'replacement', 'warranty', 'complaint', 'response'],
        'hindi': ['service', 'madad', 'help', 'vapas', 'badal']
    }
}

print("✅ Defined keywords for 5 aspects:")
for aspect, keywords in ASPECT_KEYWORDS.items():
    total_keywords = len(keywords['english']) + len(keywords['hindi'])
    print(f"   • {aspect}: {total_keywords} keywords")

# ============================================================================
# STEP 3: EXTRACT ASPECTS FROM REVIEWS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Extracting aspects from reviews...")
print("=" * 70)

def extract_aspects(text):
    """Extract which aspects are mentioned in the review"""
    if pd.isna(text):
        return []
    
    text_lower = str(text).lower()
    found_aspects = []
    
    for aspect, keywords in ASPECT_KEYWORDS.items():
        # Check both English and Hindi keywords
        all_keywords = keywords['english'] + keywords['hindi']
        
        for keyword in all_keywords:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                found_aspects.append(aspect)
                break  # Found this aspect, move to next
    
    return found_aspects if found_aspects else ['general']

print("🔄 Extracting aspects from all reviews...")
df['extracted_aspects'] = df['review_text'].apply(extract_aspects)

# Count how many reviews mention each aspect
aspect_counts = Counter()
for aspects in df['extracted_aspects']:
    aspect_counts.update(aspects)

print("\n📊 Aspects found in reviews:")
for aspect, count in aspect_counts.most_common():
    percentage = (count / len(df)) * 100
    print(f"   • {aspect}: {count:,} reviews ({percentage:.1f}%)")

# ============================================================================
# STEP 4: EXTRACT ASPECT-SPECIFIC SENTIMENT
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Analyzing sentiment for each aspect...")
print("=" * 70)

def get_aspect_sentiment(text, aspect, overall_sentiment):
    """
    Get sentiment for a specific aspect within a review.
    Uses context words around aspect keywords.
    """
    if pd.isna(text):
        return 'neutral'
    
    text_lower = str(text).lower()
    
    # Positive and negative indicator words
    positive_words = ['good', 'great', 'excellent', 'amazing', 'awesome', 'perfect',
                     'achha', 'acha', 'badhiya', 'mast', 'shandar', 'zabardast',
                     'fast', 'quick', 'love', 'best', 'nice', 'happy', 'satisfied']
    
    negative_words = ['bad', 'poor', 'terrible', 'worst', 'awful', 'horrible',
                     'kharab', 'bekaar', 'ghatiya', 'bura', 'late', 'slow',
                     'damaged', 'broken', 'disappointed', 'waste', 'rude']
    
    # Get keywords for this aspect
    aspect_keywords = ASPECT_KEYWORDS.get(aspect, {}).get('english', []) + \
                     ASPECT_KEYWORDS.get(aspect, {}).get('hindi', [])
    
    # Find sentences containing aspect keywords
    sentences = re.split(r'[.!?]', text_lower)
    aspect_sentences = [s for s in sentences 
                       if any(kw in s for kw in aspect_keywords)]
    
    if not aspect_sentences:
        return overall_sentiment  # Fallback to overall
    
    # Count positive and negative words in aspect sentences
    pos_count = sum(1 for s in aspect_sentences 
                   for word in positive_words if word in s)
    neg_count = sum(1 for s in aspect_sentences 
                   for word in negative_words if word in s)
    
    # Determine aspect sentiment
    if pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    else:
        return overall_sentiment  # Use overall sentiment as fallback

print("🔄 Analyzing aspect-level sentiment...")

# Create columns for each aspect sentiment
aspect_sentiments = {}
for aspect in ['product_quality', 'delivery', 'packaging', 'price', 'customer_service']:
    aspect_col = f'{aspect}_sentiment'
    df[aspect_col] = df.apply(
        lambda row: get_aspect_sentiment(row['review_text'], aspect, row['sentiment'])
        if aspect in row['extracted_aspects'] else None,
        axis=1
    )
    aspect_sentiments[aspect] = df[aspect_col].dropna()

print("✅ Aspect-level sentiment analysis complete!")

# ============================================================================
# STEP 5: GENERATE ASPECT ANALYSIS REPORT
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Generating aspect analysis report...")
print("=" * 70)

report_lines = []
report_lines.append("=" * 70)
report_lines.append("ASPECT-BASED SENTIMENT ANALYSIS REPORT")
report_lines.append("By: Bhavika Baddur")
report_lines.append("=" * 70)
report_lines.append("")

for aspect in ['product_quality', 'delivery', 'packaging', 'price', 'customer_service']:
    aspect_col = f'{aspect}_sentiment'
    aspect_data = df[df[aspect_col].notna()]
    
    if len(aspect_data) == 0:
        continue
    
    report_lines.append(f"\n{'=' * 70}")
    report_lines.append(f"📊 {aspect.replace('_', ' ').upper()}")
    report_lines.append(f"{'=' * 70}")
    
    # Count sentiments
    sentiment_counts = aspect_data[aspect_col].value_counts()
    total = len(aspect_data)
    
    report_lines.append(f"\nTotal mentions: {total:,} reviews")
    report_lines.append(f"\nSentiment Distribution:")
    
    for sentiment in ['positive', 'neutral', 'negative']:
        count = sentiment_counts.get(sentiment, 0)
        percentage = (count / total * 100) if total > 0 else 0
        bar = '█' * int(percentage / 2)
        report_lines.append(f"  {sentiment.capitalize():10s}: {bar:50s} {percentage:5.1f}% ({count:,})")
    
    # Sample reviews for this aspect
    report_lines.append(f"\nSample Positive Reviews:")
    positive_samples = aspect_data[aspect_data[aspect_col] == 'positive']['review_text'].head(2)
    for i, review in enumerate(positive_samples, 1):
        report_lines.append(f"  {i}. {str(review)[:100]}...")
    
    report_lines.append(f"\nSample Negative Reviews:")
    negative_samples = aspect_data[aspect_data[aspect_col] == 'negative']['review_text'].head(2)
    for i, review in enumerate(negative_samples, 1):
        report_lines.append(f"  {i}. {str(review)[:100]}...")

# Save report
ensure_directory('reports')
report_path = 'reports/aspect_analysis_report.txt'

try:
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"✅ Report saved: {report_path}")
except Exception as e:
    print(f"\n⚠️ WARNING: Could not save report: {str(e)}")

# Print summary to console
print("\n" + "=" * 70)
print("📊 ASPECT ANALYSIS SUMMARY")
print("=" * 70)

for aspect in ['product_quality', 'delivery', 'packaging', 'price', 'customer_service']:
    aspect_col = f'{aspect}_sentiment'
    aspect_data = df[df[aspect_col].notna()]
    
    if len(aspect_data) > 0:
        sentiment_counts = aspect_data[aspect_col].value_counts()
        total = len(aspect_data)
        pos_pct = (sentiment_counts.get('positive', 0) / total * 100) if total > 0 else 0
        neg_pct = (sentiment_counts.get('negative', 0) / total * 100) if total > 0 else 0
        
        status = "✅ Good" if pos_pct > 60 else "⚠️ Needs Attention" if pos_pct > 40 else "❌ Critical"
        
        print(f"\n{aspect.replace('_', ' ').title()}:")
        print(f"  Mentions: {total:,} reviews")
        print(f"  Positive: {pos_pct:.1f}% | Negative: {neg_pct:.1f}%")
        print(f"  Status: {status}")

# ============================================================================
# STEP 6: CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Creating aspect visualizations...")
print("=" * 70)

os.makedirs('reports/images', exist_ok=True)

# Visualization 1: Aspect Mention Frequency
plt.figure(figsize=(12, 6))
aspects = []
counts = []
for aspect, count in aspect_counts.most_common():
    if aspect != 'general':
        aspects.append(aspect.replace('_', '\n'))
        counts.append(count)

plt.bar(aspects, counts, color='steelblue')
plt.title('Aspect Mention Frequency in Reviews\nby Bhavika Baddur', 
          fontsize=14, weight='bold', pad=20)
plt.xlabel('Aspect', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (aspect, count) in enumerate(zip(aspects, counts)):
    plt.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig('reports/images/aspect_frequency.png', dpi=300, bbox_inches='tight')
print("✅ Saved: reports/images/aspect_frequency.png")
plt.close()

# Visualization 2: Aspect Sentiment Heatmap
print("🔄 Creating aspect sentiment heatmap...")

aspect_sentiment_data = []
for aspect in ['product_quality', 'delivery', 'packaging', 'price', 'customer_service']:
    aspect_col = f'{aspect}_sentiment'
    aspect_data = df[df[aspect_col].notna()]
    
    if len(aspect_data) > 0:
        sentiment_counts = aspect_data[aspect_col].value_counts()
        total = len(aspect_data)
        
        aspect_sentiment_data.append({
            'Aspect': aspect.replace('_', ' ').title(),
            'Positive': sentiment_counts.get('positive', 0) / total * 100,
            'Neutral': sentiment_counts.get('neutral', 0) / total * 100,
            'Negative': sentiment_counts.get('negative', 0) / total * 100
        })

sentiment_df = pd.DataFrame(aspect_sentiment_data)
sentiment_df = sentiment_df.set_index('Aspect')

plt.figure(figsize=(10, 6))
sns.heatmap(sentiment_df, annot=True, fmt='.1f', cmap='RdYlGn', 
            cbar_kws={'label': 'Percentage (%)'}, vmin=0, vmax=100)
plt.title('Aspect-Based Sentiment Distribution\nby Bhavika Baddur', 
          fontsize=14, weight='bold', pad=20)
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Aspect', fontsize=12)
plt.tight_layout()
plt.savefig('reports/images/aspect_sentiment_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Saved: reports/images/aspect_sentiment_heatmap.png")
plt.close()

# ============================================================================
# STEP 7: SAVE ENHANCED DATASET
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: Saving enhanced dataset with aspects...")
print("=" * 70)

# Save dataset with aspect information
output_path = 'processed_data/dataset_with_aspects.csv'
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"✅ Saved: {output_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("🎉 ASPECT EXTRACTION COMPLETE!")
print("=" * 70)

print(f"\n✅ RESULTS:")
print(f"   • Analyzed {len(df):,} reviews")
print(f"   • Extracted {len(aspect_counts)-1} business aspects")
print(f"   • Generated aspect-level sentiment scores")

print(f"\n📁 FILES CREATED:")
print(f"   • reports/aspect_analysis_report.txt (detailed report)")
print(f"   • reports/images/aspect_frequency.png (mention frequency)")
print(f"   • reports/images/aspect_sentiment_heatmap.png (sentiment heatmap)")
print(f"   • processed_data/dataset_with_aspects.csv (enhanced dataset)")

print(f"\n🎓 KEY INSIGHTS:")
# Find best and worst aspects
best_aspect = None
worst_aspect = None
best_score = 0
worst_score = 100

for aspect in ['product_quality', 'delivery', 'packaging', 'price', 'customer_service']:
    aspect_col = f'{aspect}_sentiment'
    aspect_data = df[df[aspect_col].notna()]
    
    if len(aspect_data) > 0:
        sentiment_counts = aspect_data[aspect_col].value_counts()
        total = len(aspect_data)
        pos_pct = (sentiment_counts.get('positive', 0) / total * 100) if total > 0 else 0
        
        if pos_pct > best_score:
            best_score = pos_pct
            best_aspect = aspect
        if pos_pct < worst_score:
            worst_score = pos_pct
            worst_aspect = aspect

if best_aspect:
    print(f"   ✅ Best performing: {best_aspect.replace('_', ' ').title()} ({best_score:.1f}% positive)")
if worst_aspect:
    print(f"   ⚠️ Needs improvement: {worst_aspect.replace('_', ' ').title()} ({worst_score:.1f}% positive)")

print(f"\n🚀 NEXT STEPS:")
print(f"   1. Review the aspect_analysis_report.txt")
print(f"   2. Check the visualizations in reports/images/")
print(f"   3. Ready for Option 3: Interactive Dashboard!")

print("\n" + "=" * 70)
print("Project by: Bhavika Baddur")
print("Multilingual E-commerce Sentiment Analysis Platform")
print("=" * 70)
