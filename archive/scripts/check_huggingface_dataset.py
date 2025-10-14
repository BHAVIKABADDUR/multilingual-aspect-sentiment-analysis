"""
Check if we need the Hugging Face Code-Mixed Sentiment Analysis Dataset
"""

import pandas as pd
import os

def check_huggingface_dataset_need():
    """Check if we need the Hugging Face dataset based on our current data"""
    
    print("=" * 60)
    print("HUGGING FACE DATASET NEED ASSESSMENT")
    print("=" * 60)
    
    # Load our current merged dataset
    merged_file = 'processed_data/merged_reviews_dataset.csv'
    
    if not os.path.exists(merged_file):
        print("❌ Merged dataset not found!")
        return
    
    df = pd.read_csv(merged_file, encoding='utf-8')
    
    print(f"\nCURRENT DATASET ANALYSIS:")
    print(f"Total Reviews: {len(df):,}")
    
    # Language analysis
    language_counts = df['language_mix'].value_counts()
    print(f"\nLanguage Distribution:")
    for lang, count in language_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {lang}: {count:,} ({percentage:.1f}%)")
    
    # Code-mixed analysis
    code_mixed_count = len(df[df['language_mix'] == 'hindi_english'])
    code_mixed_percentage = (code_mixed_count / len(df)) * 100
    
    print(f"\nCODE-MIXED ANALYSIS:")
    print(f"Hindi-English Code-mixed Reviews: {code_mixed_count:,} ({code_mixed_percentage:.1f}%)")
    
    # Sample code-mixed reviews
    code_mixed_samples = df[df['language_mix'] == 'hindi_english']['review_text'].head(5)
    print(f"\nSample Code-mixed Reviews:")
    for i, review in enumerate(code_mixed_samples, 1):
        print(f"  {i}. {review[:80]}...")
    
    # Assessment
    print(f"\nASSESSMENT:")
    
    if code_mixed_count >= 3000:
        print(f"✅ EXCELLENT: We have {code_mixed_count:,} code-mixed reviews")
        print(f"✅ This is MORE than sufficient for training and testing")
        print(f"✅ No need for additional Hugging Face dataset")
        
        recommendation = "SKIP"
        reason = f"Already have {code_mixed_count:,} code-mixed reviews (sufficient for project)"
        
    elif code_mixed_count >= 1000:
        print(f"✅ GOOD: We have {code_mixed_count:,} code-mixed reviews")
        print(f"✅ This should be sufficient for initial development")
        print(f"⚠️ Hugging Face dataset could provide additional variety")
        
        recommendation = "OPTIONAL"
        reason = f"Have {code_mixed_count:,} code-mixed reviews (good for start, HF dataset could help)"
        
    else:
        print(f"❌ INSUFFICIENT: Only {code_mixed_count:,} code-mixed reviews")
        print(f"❌ Need more code-mixed data for proper training")
        print(f"✅ Hugging Face dataset would be very helpful")
        
        recommendation = "RECOMMENDED"
        reason = f"Only have {code_mixed_count:,} code-mixed reviews (need more)"
    
    print(f"\n🎯 RECOMMENDATION: {recommendation}")
    print(f"📝 Reason: {reason}")
    
    # What the Hugging Face dataset would add
    print(f"\n📦 HUGGING FACE DATASET INFO:")
    print(f"Dataset: md-nishat-008/Code-Mixed-Sentiment-Analysis-Dataset")
    print(f"Potential Benefits:")
    print(f"  - Additional code-mixed examples")
    print(f"  - Different domain coverage")
    print(f"  - Pre-annotated sentiment labels")
    print(f"  - Research-grade quality")
    
    print(f"\nPotential Drawbacks:")
    print(f"  - Additional complexity")
    print(f"  - May duplicate existing data")
    print(f"  - Integration overhead")
    print(f"  - May not be e-commerce specific")
    
    return recommendation, code_mixed_count

def main():
    recommendation, code_mixed_count = check_huggingface_dataset_need()
    
    print(f"\n" + "=" * 60)
    print(f"FINAL DECISION")
    print("=" * 60)
    
    if recommendation == "SKIP":
        print(f"🚀 PROCEED WITHOUT Hugging Face dataset")
        print(f"✅ We have sufficient code-mixed data: {code_mixed_count:,} reviews")
        print(f"✅ Focus on sentiment analysis with current data")
        
    elif recommendation == "OPTIONAL":
        print(f"⚖️ Hugging Face dataset is OPTIONAL")
        print(f"✅ Current data sufficient for initial development")
        print(f"💡 Can add HF dataset later if needed for improvement")
        
    else:  # RECOMMENDED
        print(f"📥 RECOMMENDED to add Hugging Face dataset")
        print(f"❌ Current code-mixed data insufficient: {code_mixed_count:,} reviews")
        print(f"✅ HF dataset would significantly improve training data")
    
    print(f"\nNext Steps:")
    if recommendation == "SKIP":
        print(f"1. Start sentiment analysis with current data")
        print(f"2. Build aspect extraction model")
        print(f"3. Create dashboard prototype")
    else:
        print(f"1. Download Hugging Face dataset")
        print(f"2. Integrate with current data")
        print(f"3. Re-run data cleaning pipeline")
        print(f"4. Start sentiment analysis")

if __name__ == "__main__":
    main()
