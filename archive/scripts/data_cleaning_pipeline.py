"""
Comprehensive Data Cleaning and Merging Pipeline
Cleans and merges all e-commerce review datasets for sentiment analysis
"""

import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataCleaningPipeline:
    def __init__(self):
        self.cleaned_datasets = {}
        self.merged_dataset = None
        self.cleaning_stats = {}
        
    def load_all_datasets(self):
        """Load all available datasets"""
        print("=" * 60)
        print("LOADING ALL DATASETS")
        print("=" * 60)
        
        datasets = {}
        
        # Load synthetic dataset (reviews.csv)
        if os.path.exists('reviews.csv'):
            print("Loading synthetic reviews dataset...")
            df = pd.read_csv('reviews.csv', encoding='utf-8')
            datasets['synthetic_reviews'] = df
            print(f"  Loaded: {len(df):,} reviews")
        
        # Load Amazon VFL reviews
        if os.path.exists('amazon_vfl_reviews.csv'):
            print("Loading Amazon VFL reviews...")
            df = pd.read_csv('amazon_vfl_reviews.csv', encoding='utf-8')
            datasets['amazon_vfl'] = df
            print(f"  Loaded: {len(df):,} reviews")
        
        # Load Dataset-SA
        if os.path.exists('Dataset-SA.csv'):
            print("Loading Dataset-SA...")
            df = pd.read_csv('Dataset-SA.csv', encoding='utf-8')
            datasets['dataset_sa'] = df
            print(f"  Loaded: {len(df):,} reviews")
        
        # Load laptop reviews (with encoding fix)
        if os.path.exists('laptop.csv'):
            print("Loading laptop reviews...")
            try:
                df = pd.read_csv('laptop.csv', encoding='utf-8')
                datasets['laptop'] = df
                print(f"  Loaded: {len(df):,} reviews")
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv('laptop.csv', encoding='latin-1')
                    datasets['laptop'] = df
                    print(f"  Loaded: {len(df):,} reviews (latin-1 encoding)")
                except Exception as e:
                    print(f"  Error loading laptop.csv: {e}")
        
        # Load restaurant reviews (with encoding fix)
        if os.path.exists('restaurant.csv'):
            print("Loading restaurant reviews...")
            try:
                df = pd.read_csv('restaurant.csv', encoding='utf-8')
                datasets['restaurant'] = df
                print(f"  Loaded: {len(df):,} reviews")
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv('restaurant.csv', encoding='latin-1')
                    datasets['restaurant'] = df
                    print(f"  Loaded: {len(df):,} reviews (latin-1 encoding)")
                except Exception as e:
                    print(f"  Error loading restaurant.csv: {e}")
        
        print(f"\nTotal datasets loaded: {len(datasets)}")
        return datasets
    
    def clean_text(self, text):
        """Clean and normalize text data"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Hindi Devanagari script
        text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def standardize_rating(self, rating):
        """Standardize rating to 1-5 scale"""
        if pd.isna(rating):
            return None
        
        try:
            rating = float(rating)
            if rating < 1:
                return 1
            elif rating > 5:
                return 5
            else:
                return int(rating)
        except:
            return None
    
    def determine_sentiment_from_rating(self, rating):
        """Determine sentiment from rating"""
        if pd.isna(rating):
            return 'neutral'
        
        rating = float(rating)
        if rating >= 4:
            return 'positive'
        elif rating <= 2:
            return 'negative'
        else:
            return 'neutral'
    
    def detect_language_mix(self, text):
        """Detect if text contains Hindi-English code-mixing"""
        if pd.isna(text):
            return 'unknown'
        
        text = str(text)
        
        # Hindi patterns
        hindi_patterns = [
            r'[\u0900-\u097F]+',  # Devanagari script
            r'\b(है|था|थी|हैं|हूं|हो|कर|से|को|में|पर|के|की|का)\b',
            r'\b(अच्छा|बढ़िया|मस्त|बहुत|काफी|बिल्कुल|बेकार|भयानक|शानदार)\b'
        ]
        
        # English patterns
        english_patterns = [
            r'\b(delivery|quality|service|product|good|bad|excellent|poor|great|awesome|terrible)\b',
            r'\b(laptop|computer|phone|battery|screen|keyboard|mouse|food|restaurant)\b',
            r'\b(price|cost|expensive|cheap|worth|value|amazing|fantastic)\b'
        ]
        
        has_hindi = any(re.search(pattern, text) for pattern in hindi_patterns)
        has_english = any(re.search(pattern, text, re.IGNORECASE) for pattern in english_patterns)
        
        if has_hindi and has_english:
            return 'hindi_english'
        elif has_hindi:
            return 'hindi'
        elif has_english:
            return 'english'
        else:
            return 'other'
    
    def clean_synthetic_reviews(self, df):
        """Clean synthetic reviews dataset"""
        print("Cleaning synthetic reviews dataset...")
        
        cleaned_df = df.copy()
        
        # Standardize column names
        column_mapping = {
            'review_id': 'review_id',
            'asin': 'product_id',
            'product_name': 'product_name',
            'platform': 'platform',
            'city': 'city',
            'category': 'category',
            'review_date': 'review_date',
            'rating': 'rating',
            'sentiment': 'sentiment',
            'review_text': 'review_text',
            'language_mix': 'language_mix',
            'aspects_mentioned': 'aspects_mentioned',
            'verified_purchase': 'verified_purchase',
            'helpful_votes': 'helpful_votes',
            'review_length': 'review_length'
        }
        
        cleaned_df = cleaned_df.rename(columns=column_mapping)
        
        # Clean text
        cleaned_df['review_text'] = cleaned_df['review_text'].apply(self.clean_text)
        
        # Standardize ratings
        cleaned_df['rating'] = cleaned_df['rating'].apply(self.standardize_rating)
        
        # Add dataset source
        cleaned_df['dataset_source'] = 'synthetic'
        
        # Add review ID if missing
        if 'review_id' not in cleaned_df.columns:
            cleaned_df['review_id'] = 'SYN_' + cleaned_df.index.astype(str)
        
        self.cleaning_stats['synthetic_reviews'] = {
            'original_rows': len(df),
            'cleaned_rows': len(cleaned_df),
            'text_cleaned': True,
            'language_mix_detected': True
        }
        
        return cleaned_df
    
    def clean_amazon_vfl(self, df):
        """Clean Amazon VFL reviews dataset"""
        print("Cleaning Amazon VFL reviews dataset...")
        
        cleaned_df = df.copy()
        
        # Standardize column names
        column_mapping = {
            'asin': 'product_id',
            'name': 'product_name',
            'date': 'review_date',
            'rating': 'rating',
            'review': 'review_text'
        }
        
        cleaned_df = cleaned_df.rename(columns=column_mapping)
        
        # Clean text
        cleaned_df['review_text'] = cleaned_df['review_text'].apply(self.clean_text)
        
        # Standardize ratings
        cleaned_df['rating'] = cleaned_df['rating'].apply(self.standardize_rating)
        
        # Determine sentiment from rating
        cleaned_df['sentiment'] = cleaned_df['rating'].apply(self.determine_sentiment_from_rating)
        
        # Detect language mix
        cleaned_df['language_mix'] = cleaned_df['review_text'].apply(self.detect_language_mix)
        
        # Add missing columns with default values
        cleaned_df['platform'] = 'Amazon'
        cleaned_df['category'] = 'Beauty & Personal Care'
        cleaned_df['city'] = 'Unknown'
        cleaned_df['aspects_mentioned'] = 'product_quality'
        cleaned_df['verified_purchase'] = True
        cleaned_df['helpful_votes'] = 0
        cleaned_df['review_length'] = cleaned_df['review_text'].str.len()
        cleaned_df['dataset_source'] = 'amazon_vfl'
        
        # Generate review IDs
        cleaned_df['review_id'] = 'AMZ_' + cleaned_df.index.astype(str)
        
        self.cleaning_stats['amazon_vfl'] = {
            'original_rows': len(df),
            'cleaned_rows': len(cleaned_df),
            'text_cleaned': True,
            'language_mix_detected': True
        }
        
        return cleaned_df
    
    def clean_dataset_sa(self, df):
        """Clean Dataset-SA reviews dataset"""
        print("Cleaning Dataset-SA reviews dataset...")
        
        cleaned_df = df.copy()
        
        # Standardize column names
        column_mapping = {
            'product_name': 'product_name',
            'product_price': 'product_price',
            'Rate': 'rating',
            'Review': 'review_text',
            'Summary': 'review_summary',
            'Sentiment': 'sentiment'
        }
        
        cleaned_df = cleaned_df.rename(columns=column_mapping)
        
        # Clean text
        cleaned_df['review_text'] = cleaned_df['review_text'].apply(self.clean_text)
        cleaned_df['review_summary'] = cleaned_df['review_summary'].apply(self.clean_text)
        
        # Standardize ratings
        cleaned_df['rating'] = cleaned_df['rating'].apply(self.standardize_rating)
        
        # Standardize sentiment
        sentiment_mapping = {
            'positive': 'positive',
            'negative': 'negative',
            'neutral': 'neutral'
        }
        cleaned_df['sentiment'] = cleaned_df['sentiment'].map(sentiment_mapping).fillna('neutral')
        
        # Detect language mix
        cleaned_df['language_mix'] = cleaned_df['review_text'].apply(self.detect_language_mix)
        
        # Add missing columns with default values
        cleaned_df['platform'] = 'Unknown'
        cleaned_df['category'] = 'Electronics'
        cleaned_df['city'] = 'Unknown'
        cleaned_df['review_date'] = '2024-01-01'
        cleaned_df['aspects_mentioned'] = 'product_quality, delivery'
        cleaned_df['verified_purchase'] = True
        cleaned_df['helpful_votes'] = 0
        cleaned_df['review_length'] = cleaned_df['review_text'].str.len()
        cleaned_df['dataset_source'] = 'dataset_sa'
        
        # Generate review IDs
        cleaned_df['review_id'] = 'DSA_' + cleaned_df.index.astype(str)
        
        self.cleaning_stats['dataset_sa'] = {
            'original_rows': len(df),
            'cleaned_rows': len(cleaned_df),
            'text_cleaned': True,
            'language_mix_detected': True
        }
        
        return cleaned_df
    
    def clean_laptop_reviews(self, df):
        """Clean laptop reviews dataset"""
        print("Cleaning laptop reviews dataset...")
        
        cleaned_df = df.copy()
        
        # Standardize column names
        column_mapping = {
            'id': 'review_id',
            'sentence': 'review_text',
            '#aspect terms': 'num_aspects',
            'aspect_term': 'aspects_mentioned',
            'at_polarity': 'aspect_sentiments'
        }
        
        cleaned_df = cleaned_df.rename(columns=column_mapping)
        
        # Clean text
        cleaned_df['review_text'] = cleaned_df['review_text'].apply(self.clean_text)
        
        # Process aspects
        if 'aspects_mentioned' in cleaned_df.columns:
            cleaned_df['aspects_mentioned'] = cleaned_df['aspects_mentioned'].apply(
                lambda x: str(x).replace('[', '').replace(']', '').replace("'", '') if pd.notna(x) else ''
            )
        
        # Determine overall sentiment from aspect sentiments
        if 'aspect_sentiments' in cleaned_df.columns:
            cleaned_df['sentiment'] = cleaned_df['aspect_sentiments'].apply(
                lambda x: 'positive' if 'positive' in str(x) else 'negative' if 'negative' in str(x) else 'neutral'
            )
        else:
            cleaned_df['sentiment'] = 'neutral'
        
        # Standardize ratings (estimate from sentiment)
        rating_mapping = {'positive': 4, 'negative': 2, 'neutral': 3}
        cleaned_df['rating'] = cleaned_df['sentiment'].map(rating_mapping)
        
        # Detect language mix
        cleaned_df['language_mix'] = cleaned_df['review_text'].apply(self.detect_language_mix)
        
        # Add missing columns with default values
        cleaned_df['platform'] = 'Unknown'
        cleaned_df['category'] = 'Electronics'
        cleaned_df['city'] = 'Unknown'
        cleaned_df['review_date'] = '2024-01-01'
        cleaned_df['verified_purchase'] = True
        cleaned_df['helpful_votes'] = 0
        cleaned_df['review_length'] = cleaned_df['review_text'].str.len()
        cleaned_df['dataset_source'] = 'laptop'
        
        self.cleaning_stats['laptop'] = {
            'original_rows': len(df),
            'cleaned_rows': len(cleaned_df),
            'text_cleaned': True,
            'language_mix_detected': True
        }
        
        return cleaned_df
    
    def clean_restaurant_reviews(self, df):
        """Clean restaurant reviews dataset"""
        print("Cleaning restaurant reviews dataset...")
        
        cleaned_df = df.copy()
        
        # Standardize column names
        column_mapping = {
            'id': 'review_id',
            'sentence': 'review_text',
            'num_aspect_terms': 'num_aspects',
            'aspect_terms': 'aspects_mentioned',
            'at_polarity': 'aspect_sentiments'
        }
        
        cleaned_df = cleaned_df.rename(columns=column_mapping)
        
        # Clean text
        cleaned_df['review_text'] = cleaned_df['review_text'].apply(self.clean_text)
        
        # Process aspects
        if 'aspects_mentioned' in cleaned_df.columns:
            cleaned_df['aspects_mentioned'] = cleaned_df['aspects_mentioned'].apply(
                lambda x: str(x).replace('[', '').replace(']', '').replace("'", '') if pd.notna(x) else ''
            )
        
        # Determine overall sentiment from aspect sentiments
        if 'aspect_sentiments' in cleaned_df.columns:
            cleaned_df['sentiment'] = cleaned_df['aspect_sentiments'].apply(
                lambda x: 'positive' if 'positive' in str(x) else 'negative' if 'negative' in str(x) else 'neutral'
            )
        else:
            cleaned_df['sentiment'] = 'neutral'
        
        # Standardize ratings (estimate from sentiment)
        rating_mapping = {'positive': 4, 'negative': 2, 'neutral': 3}
        cleaned_df['rating'] = cleaned_df['sentiment'].map(rating_mapping)
        
        # Detect language mix
        cleaned_df['language_mix'] = cleaned_df['review_text'].apply(self.detect_language_mix)
        
        # Add missing columns with default values
        cleaned_df['platform'] = 'Unknown'
        cleaned_df['category'] = 'Food & Beverages'
        cleaned_df['city'] = 'Unknown'
        cleaned_df['review_date'] = '2024-01-01'
        cleaned_df['verified_purchase'] = True
        cleaned_df['helpful_votes'] = 0
        cleaned_df['review_length'] = cleaned_df['review_text'].str.len()
        cleaned_df['dataset_source'] = 'restaurant'
        
        self.cleaning_stats['restaurant'] = {
            'original_rows': len(df),
            'cleaned_rows': len(cleaned_df),
            'text_cleaned': True,
            'language_mix_detected': True
        }
        
        return cleaned_df
    
    def clean_all_datasets(self, datasets):
        """Clean all datasets"""
        print("\n" + "=" * 60)
        print("CLEANING ALL DATASETS")
        print("=" * 60)
        
        for name, df in datasets.items():
            if name == 'synthetic_reviews':
                self.cleaned_datasets[name] = self.clean_synthetic_reviews(df)
            elif name == 'amazon_vfl':
                self.cleaned_datasets[name] = self.clean_amazon_vfl(df)
            elif name == 'dataset_sa':
                self.cleaned_datasets[name] = self.clean_dataset_sa(df)
            elif name == 'laptop':
                self.cleaned_datasets[name] = self.clean_laptop_reviews(df)
            elif name == 'restaurant':
                self.cleaned_datasets[name] = self.clean_restaurant_reviews(df)
        
        print(f"\nCleaned {len(self.cleaned_datasets)} datasets successfully!")
        
        # Print cleaning statistics
        print("\nCleaning Statistics:")
        for name, stats in self.cleaning_stats.items():
            print(f"  {name}: {stats['original_rows']:,} -> {stats['cleaned_rows']:,} rows")
    
    def merge_datasets(self):
        """Merge all cleaned datasets into one unified dataset"""
        print("\n" + "=" * 60)
        print("MERGING ALL DATASETS")
        print("=" * 60)
        
        if not self.cleaned_datasets:
            print("No cleaned datasets to merge!")
            return None
        
        # Define common columns for all datasets
        common_columns = [
            'review_id', 'product_id', 'product_name', 'platform', 'category',
            'city', 'review_date', 'rating', 'sentiment', 'review_text',
            'language_mix', 'aspects_mentioned', 'verified_purchase',
            'helpful_votes', 'review_length', 'dataset_source'
        ]
        
        # Prepare each dataset with common columns
        prepared_datasets = []
        
        for name, df in self.cleaned_datasets.items():
            print(f"Preparing {name} dataset...")
            
            # Ensure all common columns exist
            for col in common_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Select only common columns
            prepared_df = df[common_columns].copy()
            
            # Add dataset info
            prepared_df['dataset_source'] = name
            
            prepared_datasets.append(prepared_df)
            print(f"  Prepared: {len(prepared_df):,} rows")
        
        # Merge all datasets
        print("\nMerging all datasets...")
        merged_df = pd.concat(prepared_datasets, ignore_index=True)
        
        # Remove duplicates based on review text
        print("Removing duplicate reviews...")
        initial_count = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=['review_text'], keep='first')
        final_count = len(merged_df)
        duplicates_removed = initial_count - final_count
        
        print(f"  Removed {duplicates_removed:,} duplicate reviews")
        print(f"  Final merged dataset: {final_count:,} reviews")
        
        self.merged_dataset = merged_df
        return merged_df
    
    def save_cleaned_datasets(self):
        """Save cleaned and merged datasets"""
        print("\n" + "=" * 60)
        print("SAVING CLEANED DATASETS")
        print("=" * 60)
        
        # Create cleaned_datasets directory
        os.makedirs('cleaned_datasets', exist_ok=True)
        
        # Save individual cleaned datasets
        for name, df in self.cleaned_datasets.items():
            filename = f'cleaned_datasets/{name}_cleaned.csv'
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"Saved {name}: {len(df):,} reviews -> {filename}")
        
        # Save merged dataset
        if self.merged_dataset is not None:
            filename = 'cleaned_datasets/merged_reviews_dataset.csv'
            self.merged_dataset.to_csv(filename, index=False, encoding='utf-8')
            print(f"Saved merged dataset: {len(self.merged_dataset):,} reviews -> {filename}")
            
            # Save a sample for quick testing
            sample_filename = 'cleaned_datasets/sample_merged_dataset.csv'
            sample_df = self.merged_dataset.sample(n=min(1000, len(self.merged_dataset)), random_state=42)
            sample_df.to_csv(sample_filename, index=False, encoding='utf-8')
            print(f"Saved sample dataset: {len(sample_df):,} reviews -> {sample_filename}")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if self.merged_dataset is None:
            print("No merged dataset available for summary!")
            return
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE DATASET SUMMARY REPORT")
        print("=" * 60)
        
        df = self.merged_dataset
        
        print(f"\nTOTAL REVIEWS: {len(df):,}")
        
        # Dataset source distribution
        print(f"\nDataset Source Distribution:")
        source_counts = df['dataset_source'].value_counts()
        for source, count in source_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {source}: {count:,} ({percentage:.1f}%)")
        
        # Sentiment distribution
        print(f"\nSentiment Distribution:")
        sentiment_counts = df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {sentiment}: {count:,} ({percentage:.1f}%)")
        
        # Language mix distribution
        print(f"\nLanguage Mix Distribution:")
        language_counts = df['language_mix'].value_counts()
        for language, count in language_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {language}: {count:,} ({percentage:.1f}%)")
        
        # Platform distribution
        print(f"\nPlatform Distribution:")
        platform_counts = df['platform'].value_counts()
        for platform, count in platform_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {platform}: {count:,} ({percentage:.1f}%)")
        
        # Category distribution
        print(f"\nCategory Distribution:")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {category}: {count:,} ({percentage:.1f}%)")
        
        # Rating distribution
        print(f"\nRating Distribution:")
        rating_counts = df['rating'].value_counts().sort_index()
        for rating, count in rating_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {rating} stars: {count:,} ({percentage:.1f}%)")
        
        # Average review length
        avg_length = df['review_length'].mean()
        print(f"\nAverage Review Length: {avg_length:.1f} characters")
        
        # Code-mixed examples
        code_mixed_reviews = df[df['language_mix'] == 'hindi_english']['review_text'].head(5)
        if len(code_mixed_reviews) > 0:
            print(f"\nSample Code-Mixed Reviews:")
            for i, review in enumerate(code_mixed_reviews, 1):
                print(f"  {i}. {review[:100]}...")

def main():
    """Main function to run the complete data cleaning pipeline"""
    
    pipeline = DataCleaningPipeline()
    
    # Step 1: Load all datasets
    datasets = pipeline.load_all_datasets()
    
    if not datasets:
        print("No datasets found to clean!")
        return
    
    # Step 2: Clean all datasets
    pipeline.clean_all_datasets(datasets)
    
    # Step 3: Merge all datasets
    merged_df = pipeline.merge_datasets()
    
    # Step 4: Save cleaned datasets
    pipeline.save_cleaned_datasets()
    
    # Step 5: Generate summary report
    pipeline.generate_summary_report()
    
    print(f"\n" + "=" * 60)
    print("DATA CLEANING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return pipeline

if __name__ == "__main__":
    pipeline = main()
