"""
Configuration File
==================
Central configuration for all paths and parameters used in the project.

Author: Bhavika Baddur
Project: E-commerce Multilingual Sentiment Analysis
"""

import os

# ============================================================================
# DATA PATHS
# ============================================================================

# Main datasets
MAIN_DATASET = 'processed_data/enhanced_final_dataset_cleaned.csv'
ASPECT_DATASET = 'processed_data/dataset_with_aspects.csv'
ORIGINAL_DATASET = 'processed_data/enhanced_final_dataset.csv'
SAMPLE_DATASET = 'processed_data/enhanced_final_dataset_sample_2000.csv'

# Raw data directory
RAW_DATA_DIR = 'raw_data/'

# ============================================================================
# MODEL PATHS
# ============================================================================

# Trained models
SENTIMENT_MODEL = 'models/sentiment_model.pkl'
VECTORIZER = 'models/tfidf_vectorizer.pkl'

# Model directory
MODELS_DIR = 'models/'

# ============================================================================
# REPORT PATHS
# ============================================================================

# Reports directory
REPORTS_DIR = 'reports/'
IMAGES_DIR = 'reports/images/'

# Report files
DATA_SUMMARY = 'reports/data_summary.txt'
QUALITY_REPORT = 'reports/quality_report.txt'
ASPECT_REPORT = 'reports/aspect_analysis_report.txt'

# Visualization files
CONFUSION_MATRIX = 'reports/images/confusion_matrix.png'
RATING_DISTRIBUTION = 'reports/images/rating_distribution.png'
RATING_PIE_CHART = 'reports/images/rating_pie_chart.png'
ASPECT_FREQUENCY = 'reports/images/aspect_frequency.png'
ASPECT_HEATMAP = 'reports/images/aspect_sentiment_heatmap.png'

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Random Forest parameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 50
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2
RF_RANDOM_STATE = 42

# TF-IDF parameters
TFIDF_MAX_FEATURES = 8000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.75

# Data balancing
BALANCE_RATIO = 0.7  # 70% of majority class

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# ASPECT KEYWORDS
# ============================================================================

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

# ============================================================================
# BUSINESS THRESHOLDS
# ============================================================================

# Sentiment thresholds for classification
GOOD_THRESHOLD = 0.60  # >= 60% positive is good
ATTENTION_THRESHOLD = 0.40  # < 60% but >= 40% needs attention
CRITICAL_THRESHOLD = 0.40  # < 40% is critical

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Figure sizes
FIGURE_SIZE_LARGE = (12, 6)
FIGURE_SIZE_MEDIUM = (10, 6)
FIGURE_SIZE_SMALL = (8, 5)

# DPI for saving images
IMAGE_DPI = 300

# Color schemes
COLOR_POSITIVE = '#2ecc71'  # Green
COLOR_NEUTRAL = '#f39c12'   # Orange
COLOR_NEGATIVE = '#e74c3c'  # Red

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)

def get_absolute_path(relative_path):
    """Get absolute path from relative path"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)

# ============================================================================
# INITIALIZE DIRECTORIES
# ============================================================================

def initialize_directories():
    """Create all necessary directories"""
    directories = [
        MODELS_DIR,
        REPORTS_DIR,
        IMAGES_DIR,
        'logs/'
    ]
    for directory in directories:
        ensure_directory_exists(directory)

# Auto-initialize when config is imported
initialize_directories()
