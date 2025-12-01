# Dataset Backup Folder

## Purpose
This folder contains backup datasets that are NOT used for training.

## Primary Dataset (for training)
- **Location:** `processed_data/balanced_dataset.csv`
- **Rows:** 13,923
- **Use:** MuRIL training (optimized, balanced, cleaned)

## Backup Datasets

### 1. enhanced_final_dataset_cleaned.csv
- **Purpose:** Original cleaned dataset (20,000 reviews)
- **Status:** Imbalanced (61% positive, 26% negative, 13% neutral)
- **Use Case:** Reference, comparison, or re-balancing if needed

### 2. dataset_with_aspects.csv
- **Purpose:** Dataset with extracted aspect columns
- **Use Case:** Aspect-based analysis, future enhancements

### 3. muril_ready_dataset.csv
- **Purpose:** Minimal version (text + sentiment + ID only)
- **Use Case:** Quick loading, minimal memory usage

## Important Notes
- ✅ **Use `balanced_dataset.csv` for all training**
- ✅ These backups are for reference only
- ✅ Do NOT upload backups to Colab (wastes time/space)
- ✅ Keep backups for project documentation

## Last Updated
2025-11-02 12:50:41

Author: Bhavika Baddur
