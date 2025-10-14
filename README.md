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
│   ├── enhanced_final_dataset.csv            # Canonical dataset (20,000 rows)
│   └── enhanced_final_dataset_sample_2000.csv# Sample for quick iteration
│
├── 📂 scripts/                               # All project scripts
│   ├── build_final_dataset.py                # Build pipeline + synthetic generator
│   ├── 01_explore_data.py                    # Generate reports/data_summary.txt
│   ├── 02_check_quality.py                   # Generate reports/quality_report.txt
│   └── 03_first_visualization.py             # Save rating charts to reports/images
│
├── 📂 reports/                               # Generated reports and images
│   ├── data_summary.txt
│   ├── quality_report.txt
│   └── images/
│       ├── rating_distribution.png
│       └── rating_pie_chart.png
│
├── 📂 docs/                                  # Documentation
│   ├── Complete Project Description
│   ├── project_organization.md
│   └── huggingface_dataset_assessment.md
│
└── 📄 requirements.txt                       # Python dependencies
```

## 🚀 Key Features

### ✅ Completed:
- Data collection and cleaning
- Domain-coherent code-mixed synthetic generation (platform/category/product aligned)
- Unified build pipeline to `processed_data/enhanced_final_dataset.csv`
- Neutral IDs in data (no synthetic markers in `review_id`/`product_id`)
- First visualizations: rating bar/pie charts

### 🎯 Ready for Development:
- **Sentiment Analysis**: Ready to implement on 12,147 reviews
- **Aspect Extraction**: Focus on product quality, delivery, packaging, price, customer service
- **Dashboard Development**: Use sample dataset (1,000 reviews) for prototyping
- **Model Training**: Train on full merged dataset

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
```

## 📊 Usage

### Primary Dataset:
```python
import pandas as pd

df = pd.read_csv('processed_data/enhanced_final_dataset.csv')
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

- **Data Collection**: ✅ Complete
- **Data Cleaning**: ✅ Complete  
- **File Organization**: ✅ Complete
- **Ready for Analysis**: ✅ Yes
- **Sentiment Analysis**: 🔄 Next Phase
- **Aspect Extraction**: 🔄 Next Phase
- **Dashboard**: 🔄 Next Phase

## 📞 Contact

This project is part of a comprehensive e-commerce sentiment analysis system for Indian markets, focusing on multilingual customer intelligence and actionable business insights.

---

**Final Dataset**: 20,000 reviews (13k code-mixed, 7k English)
**Language Coverage**: English + Hindi-English (code-mixed)
