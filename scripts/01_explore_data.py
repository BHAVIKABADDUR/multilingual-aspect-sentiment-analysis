import pandas as pd

# 1. Load the final reviews file
data_path = 'processed_data/enhanced_final_dataset.csv'
df = pd.read_csv(data_path, encoding='utf-8')

with open('data_summary.txt', 'w', encoding='utf-8') as f:
    def write_and_print(msg):
        print(msg)
        f.write(msg + '\n')

    # 2. Show total number of reviews
    write_and_print("========== TOTAL REVIEWS ==========")
    write_and_print(f"Total reviews: {len(df):,}")
    write_and_print("")

    # 2. Show column names
    write_and_print("========== COLUMN NAMES ==========")
    write_and_print(", ".join(df.columns))
    write_and_print("")

    # 2. First 10 reviews (pretty format)
    write_and_print("========== FIRST 10 REVIEWS ==========")
    subset_cols = [c for c in ['review_id','review_text','rating','sentiment','review_date','platform','aspects_mentioned'] if c in df.columns]
    for idx, row in df.head(10).iterrows():
        write_and_print(f"--- Review {idx+1} ---")
        for col in subset_cols:
            # Shorten lengthy review text display
            val = str(row[col])
            if col == 'review_text' and len(val) > 140:
                val = val[:140] + '...'
            write_and_print(f"  {col}: {val}")
        write_and_print("")
    write_and_print("")

    # 2. Number of reviews for each star rating (1-5)
    write_and_print("========== REVIEWS PER STAR RATING ==========")
    rating_counts = df['rating'].value_counts().sort_index()
    for star, count in rating_counts.items():
        write_and_print(f"{star} star: {count}")
    write_and_print("")

    # 2. Date range of reviews
    write_and_print("========== DATE RANGE ==========")
    if 'review_date' in df.columns:
        try:
            parsed_dates = pd.to_datetime(df['review_date'], errors='coerce')
            min_date = parsed_dates.min()
            max_date = parsed_dates.max()
            write_and_print(f"From: {min_date.date()}  To: {max_date.date()}")
        except Exception as e:
            write_and_print(f"Date parsing error: {e}")
    else:
        write_and_print("No 'review_date' column found.")
    write_and_print("")

# End of script
