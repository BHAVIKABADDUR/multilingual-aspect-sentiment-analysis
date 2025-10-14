# E-commerce Sentiment Analysis Project - File Organization

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
│   ├── enhanced_final_dataset.csv            # Canonical dataset (20,000)
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

## 🎯 Key Files for Development

- `processed_data/enhanced_final_dataset.csv` — Main dataset for analysis (20,000)
- `processed_data/enhanced_final_dataset_sample_2000.csv` — Sample for quick iteration
- `scripts/01_explore_data.py` — Summaries → `reports/data_summary.txt`
- `scripts/02_check_quality.py` — Quality checks → `reports/quality_report.txt`
- `scripts/03_first_visualization.py` — Charts → `reports/images/`
