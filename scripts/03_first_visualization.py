import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed dataset
# Use the cleaned dataset with proper sentiment labels
DATA_PATH = 'processed_data/enhanced_final_dataset_cleaned.csv'


def save_rating_bar_chart(df: pd.DataFrame, out_path: str) -> None:
    """Create and save a professional bar chart of rating distribution (1-5)."""
    # Compute counts for ratings 1..5, ensure ordering
    rating_counts = df['rating'].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)

    # Style
    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6), dpi=150)
    bar_colors = sns.color_palette('viridis', n_colors=5)

    # Plot
    ax = sns.barplot(x=rating_counts.index, y=rating_counts.values, palette=bar_colors)
    ax.set_title('Distribution of Review Ratings', fontsize=16, weight='bold')
    ax.set_xlabel('Star Rating (1-5)', fontsize=12)
    ax.set_ylabel('Number of Reviews', fontsize=12)
    ax.set_xticklabels([str(i) for i in rating_counts.index], fontsize=11)
    ax.tick_params(axis='y', labelsize=10)

    # Annotate each bar with the count value
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height):,}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def save_rating_pie_chart(df: pd.DataFrame, out_path: str) -> None:
    """Create and save a professional pie chart for rating percentage breakdown."""
    counts = df['rating'].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    labels = [f'{r}-Star' for r in counts.index]
    colors = sns.color_palette('viridis', n_colors=5)

    plt.figure(figsize=(8, 8), dpi=150)
    wedges, texts, autotexts = plt.pie(
        counts.values,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10}
    )
    plt.title('Rating Breakdown (Percentage)', fontsize=16, weight='bold')
    plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def main():
    # Read dataset
    df = pd.read_csv(DATA_PATH, encoding='utf-8')

    # Ensure ratings are numeric and in 1..5
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0).astype(int)
    df = df[df['rating'].isin([1, 2, 3, 4, 5])]

    # Create output directory
    import os
    os.makedirs('reports/images', exist_ok=True)

    # Save bar chart
    save_rating_bar_chart(df, 'reports/images/rating_distribution.png')

    # Save pie chart
    save_rating_pie_chart(df, 'reports/images/rating_pie_chart.png')

    print('✅ Saved rating_distribution.png and rating_pie_chart.png to reports/images/')


if __name__ == '__main__':
    main()


