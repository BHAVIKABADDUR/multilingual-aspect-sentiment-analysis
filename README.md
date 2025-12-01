# E-commerce Sentiment Analysis Project

## 🎯 Project Overview
Multilingual Customer Intelligence Platform for Indian E-commerce - A comprehensive sentiment analysis system that processes customer reviews in English, Hindi, and code-mixed (Hinglish) languages to provide actionable business insights.

## 📊 Dataset Summary
- **Total Reviews (current build)**: 20,000 reviews (13k code-mixed + 7k English)
- **Platforms**: Amazon, Flipkart, Myntra, Nykaa, Swiggy, Zomato
- **Categories**: Electronics, Home, Kitchen, Fashion, Beauty, Food

## 📁 Project Structure

```
ecommerce-sentiment-project/
├── 📂 raw_data/                              # Original, unprocessed datasets
│   ├── amazon_vfl_reviews.csv
│   ├── Dataset-SA.csv
│   ├── laptop.csv
│   └── restaurant.csv
│
├── 📂 processed_data/                        # Final datasets for analysis
│   ├── enhanced_final_dataset_cleaned.csv    # ⭐ Main dataset (20,000 rows, cleaned)
│   ├── dataset_with_aspects.csv              # ⭐ With aspect columns (latest)
│   ├── enhanced_final_dataset.csv            # Original (backup)
│   └── enhanced_final_dataset_sample_2000.csv# Sample for quick iteration
│
├── 📂 scripts/                               # All project scripts
│   ├── build_final_dataset.py                # Build pipeline + synthetic generator
│   ├── 01_explore_data.py                    # Generate reports/data_summary.txt
│   ├── 02_check_quality.py                   # Generate reports/quality_report.txt
│   ├── 03_first_visualization.py             # Save rating charts to reports/images
│   ├── 04_simple_sentiment_analysis.py       # ⭐ Sentiment model (76.71% accuracy)
│   └── 05_aspect_extraction.py               # ⭐ Aspect analysis (5 aspects)
│
├── 📂 reports/                               # Generated reports and images
│   ├── data_summary.txt
│   ├── quality_report.txt
│   ├── aspect_analysis_report.txt            # ⭐ Aspect insights
│   └── images/
│       ├── confusion_matrix.png              # Sentiment results (76.71%)
│       ├── rating_distribution.png
│       ├── rating_pie_chart.png
│       ├── aspect_frequency.png              # ⭐ Aspect mentions
│       └── aspect_sentiment_heatmap.png      # ⭐ Aspect sentiment
│
├── 📂 docs/                                  # 📚 All Documentation
│   ├── Complete Project Description          # Full project guide
│   ├── project_organization.md
│   ├── huggingface_dataset_assessment.md
│   ├── EXECUTIVE_SUMMARY.md                  # ⭐ Business insights & ROI
│   ├── PROJECT_STATUS.md                     # ⭐ Current achievements
│   ├── PROJECT_CHECKLIST.md                  # ⭐ Task tracking
│   ├── IMPROVEMENT_SUGGESTIONS.md            # ⭐ 19 improvement ideas
│   ├── MODEL_PERFORMANCE_LOG.md              # ⭐ Model tracking
│   ├── QUICK_START.md                        # Quick start guide
│   └── HOW_TO_RUN_BEGINNER_GUIDE.md         # Detailed beginner guide
│
├── 📂 models/                                # Trained models
│   ├── sentiment_model.pkl                   # Random Forest (76.71% accuracy)
│   └── tfidf_vectorizer.pkl                  # Text processor
│
├── 📄 README.md                              # This file
├── 📄 config.py                              # ⭐ Centralized configuration
├── 📄 .gitignore                             # ⭐ Git best practices
├── 📄 requirements.txt                       # Python dependencies
└── 📄 model_performance_log.csv              # ⭐ Performance tracking data
```

## 🚀 Key Features

### ✅ Phase 1: Data Preparation (COMPLETE)
- ✅ 20,000 reviews collected and cleaned
- ✅ Domain-coherent code-mixed synthetic generation
- ✅ Unified build pipeline
- ✅ Quality assessment and visualizations

### ✅ Phase 2: Sentiment Analysis (COMPLETE)
- ✅ **76.71% accuracy** achieved with Random Forest
- ✅ Smart data balancing (70% strategy)
- ✅ 8,000 TF-IDF features with bigrams
- ✅ Confusion matrix and model evaluation
- ✅ Production-ready model saved

### ✅ Phase 3: Aspect Extraction (COMPLETE)
- ✅ 5 business aspects identified: Product Quality, Delivery, Packaging, Price, Customer Service
- ✅ Aspect-level sentiment analysis
- ✅ Critical insights: Customer Service (36.8% positive) & Packaging (39.3% positive) need improvement
- ✅ Product Quality (69.9% positive) & Price (61.7% positive) are strengths

### 🎯 Phase 4: Interactive Dashboard (NEXT)
- Build Streamlit web app
- Real-time sentiment & aspect analysis
- Interactive filters and visualizations
- Custom review analyzer

## 📋 Sample Code-Mixed Reviews

1. "Fantastic product Quality amazing hai aur delivery on time Price reasonable hai Packaging outstanding hai"
2. "Perfect product Quality great hai aur delivery fast hui Good price quality ratio Support excellent hai"
3. "Wonderful product hai Packaging good hai aur quality excellent Price affordable hai Customer care outstanding"

## 🛠️ Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Build the final dataset (writes to processed_data/)
python scripts/build_final_dataset.py

# Generate reports
python scripts/01_explore_data.py
python scripts/02_check_quality.py

# Create rating charts (saved to reports/images/)
python scripts/03_first_visualization.py

# Run sentiment analysis (BEGINNER-FRIENDLY!)
python scripts/04_simple_sentiment_analysis.py

# Run aspect extraction
python scripts/05_aspect_extraction.py
```

## 📊 Usage

### Primary Dataset:
```python
import pandas as pd

# Main cleaned dataset
df = pd.read_csv('processed_data/enhanced_final_dataset_cleaned.csv')

# Dataset with aspect columns
df_aspects = pd.read_csv('processed_data/dataset_with_aspects.csv')

# Sample for testing
sample_df = pd.read_csv('processed_data/enhanced_final_dataset_sample_2000.csv')
```

### Key Columns:
- `review_text`: The actual review content
- `language_mix`: Language type (english, hindi_english, hindi, other)
- `sentiment`: Sentiment label (positive, negative, neutral)
- `rating`: Star rating (1-5)
- `aspects_mentioned`: Business aspects mentioned
- `platform`: E-commerce platform (Amazon, Flipkart, etc.)
- `category`: Product category

## 🎯 Next Steps

1. **Sentiment Analysis Implementation**
2. **Aspect Extraction Model Development**
3. **Dashboard Creation**
4. **Model Training and Evaluation**
5. **Business Intelligence Features**

## 📈 Project Status

- **Data Collection**: ✅ Complete (20,000 reviews)
- **Data Cleaning**: ✅ Complete
- **File Organization**: ✅ Complete
- **Sentiment Analysis**: ✅ Complete (76.71% accuracy)
- **Aspect Extraction**: ✅ Complete (5 aspects analyzed)
- **Dashboard**: 🔄 Ready to Build

### 🎯 Current Achievements:
- **Model Accuracy**: 76.71% (Random Forest)
- **Aspects Analyzed**: Product Quality, Delivery, Packaging, Price, Customer Service
- **Key Insight**: Customer Service (36.8% positive) and Packaging (39.3% positive) are critical areas needing improvement
- **Strengths**: Product Quality (69.9% positive) and Price (61.7% positive)

## 🎓 For Beginners

New to machine learning? Start here:

1. **Read the guide**: `docs/HOW_TO_RUN_BEGINNER_GUIDE.md`
2. **Quick start**: `docs/QUICK_START.md`
3. **Run your first model**: `python scripts/04_simple_sentiment_analysis.py`
4. **Get results in 5 minutes** with easy-to-understand explanations!

This beginner script will:
- Train a sentiment analysis model on your 20,000 reviews
- Achieve **76.71% accuracy** (excellent for multilingual text!)
- Create easy-to-understand visualizations
- Explain every step in simple language
- Generate confusion matrix and performance reports

## 📞 Contact

This project is part of a comprehensive e-commerce sentiment analysis system for Indian markets, focusing on multilingual customer intelligence and actionable business insights.

---

**Final Dataset**: 20,000 reviews (13k code-mixed, 7k English)
**Language Coverage**: English + Hindi-English (code-mixed)
