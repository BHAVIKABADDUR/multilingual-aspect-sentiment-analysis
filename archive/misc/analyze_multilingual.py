import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the enhanced dataset
input_file = os.path.join('processed_data', 'enhanced_final_dataset.csv')

print(f"Loading dataset from {input_file}...")
df = pd.read_csv(input_file)

# Print dataset statistics
total_reviews = len(df)
print(f"Total reviews in dataset: {total_reviews}")

# Language distribution
print("\nLanguage distribution:")
language_counts = df['language_mix'].value_counts()
print(language_counts)
print(f"\nPercentage of each language:")
language_percentage = (language_counts / total_reviews * 100).round(2)
print(language_percentage)

# Sentiment distribution by language
print("\nSentiment distribution by language:")
sentiment_by_language = df.groupby(['language_mix', 'sentiment']).size().unstack()
print(sentiment_by_language)

# Calculate percentage of each sentiment within each language
sentiment_percentage = sentiment_by_language.div(sentiment_by_language.sum(axis=1), axis=0) * 100
print("\nPercentage of sentiments within each language:")
print(sentiment_percentage.round(2))

# Create output directory for plots if it doesn't exist
plots_dir = 'analysis_plots'
os.makedirs(plots_dir, exist_ok=True)

# Plot language distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=language_counts.index, y=language_counts.values)
plt.title('Language Distribution in Dataset')
plt.xlabel('Language')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'language_distribution.png'))

# Plot sentiment distribution by language
plt.figure(figsize=(12, 8))
sentiment_by_language.plot(kind='bar')
plt.title('Sentiment Distribution by Language')
plt.xlabel('Language')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'sentiment_by_language.png'))

# Plot percentage of sentiments within each language
plt.figure(figsize=(12, 8))
sentiment_percentage.plot(kind='bar', stacked=True)
plt.title('Percentage of Sentiments within Each Language')
plt.xlabel('Language')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'sentiment_percentage_by_language.png'))

print(f"\nPlots saved to {plots_dir} directory")

# Sample reviews from each language category
print("\nSample reviews from each language category:")
for language in df['language_mix'].unique():
    print(f"\n--- {language.upper()} SAMPLE REVIEWS ---")
    samples = df[df['language_mix'] == language].sample(min(3, len(df[df['language_mix'] == language])), random_state=42)
    for i, (_, row) in enumerate(samples.iterrows()):
        print(f"{i+1}. Sentiment: {row['sentiment']}")
        print(f"   Review: {row['review_text']}")
        print()

# Save analysis results to a text file
with open(os.path.join(plots_dir, 'language_analysis_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(f"Total reviews in dataset: {total_reviews}\n\n")
    f.write("Language distribution:\n")
    f.write(str(language_counts) + "\n\n")
    f.write("Percentage of each language:\n")
    f.write(str(language_percentage) + "\n\n")
    f.write("Sentiment distribution by language:\n")
    f.write(str(sentiment_by_language) + "\n\n")
    f.write("Percentage of sentiments within each language:\n")
    f.write(str(sentiment_percentage.round(2)) + "\n")

print(f"\nAnalysis summary saved to {os.path.join(plots_dir, 'language_analysis_summary.txt')}")