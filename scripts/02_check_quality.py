import pandas as pd
import re

# Load the cleaned final dataset
csv_path = 'processed_data/enhanced_final_dataset.csv'
df = pd.read_csv(csv_path, encoding='utf-8')

total = len(df)
with open('quality_report.txt', 'w', encoding='utf-8') as f:
    def write(msg):
        print(msg)
        f.write(str(msg)+'\n')
    def section(title):
        write(f"\n{'='*10} {title} {'='*10}")

    section('DATASET QUALITY ANALYSIS')
    write(f"Total reviews: {total}")

    # 1. Missing values
    section('MISSING VALUES')
    missing_reviews = df['review_text'].isna() | (df['review_text'].astype(str).str.strip()=='')
    missing_ratings = df['rating'].isna()
    write(f"Empty review_text: {missing_reviews.sum()} ({missing_reviews.sum()/total:.2%})")
    write(f"Missing rating: {missing_ratings.sum()} ({missing_ratings.sum()/total:.2%})")
    write("These rows should be dropped in analysis/modeling.")

    # 2. Very short reviews <5 words
    section('VERY SHORT REVIEWS (<5 WORDS)')
    word_counts = df['review_text'].astype(str).apply(lambda x: len(x.split()))
    short_mask = word_counts < 5
    write(f"Very short reviews: {short_mask.sum()} ({short_mask.sum()/total:.2%})")
    write("Short reviews are less informative and may hurt model performance.")

    # 3. Duplicates: same review_text
    section('DUPLICATE REVIEWS')
    duplicate_mask = df.duplicated('review_text', keep=False)
    ndup = duplicate_mask.sum()
    write(f"Reviews with same text: {ndup} ({ndup/total:.2%})")
    write("Duplicates may indicate spam, copy-pasting, etc.")

    # 4. Language mix
    section('LANGUAGE MIX DETECTION')
    hindi_words = ['hai','tha','achha','bahut','lekin','nahi','kya','kar','kaise','ke','mein','se','par']
    def has_hindi(text):
        t = str(text).lower()
        return any(word in t for word in hindi_words)
    def is_ascii(text):
        try:
            text.encode('ascii')
            return True
        except:
            return False
    hindi_like = df['review_text'].apply(has_hindi)
    pure_english = ~hindi_like & df['review_text'].apply(is_ascii)
    code_mixed = hindi_like
    write(f"Contains Hindi word: {hindi_like.sum()} ({hindi_like.sum()/total:.2%}) (flagged as code-mixed)")
    write(f"Likely pure English: {pure_english.sum()} ({pure_english.sum()/total:.2%})")
    write(f"Other (not matching): {total-hindi_like.sum()-pure_english.sum()} ({(total-hindi_like.sum()-pure_english.sum())/total:.2%})")
    write("Detection uses common Hindi tokens in Latin script; ASCII-only and no Hindi token = likely English.")

    # 5. Examples
    section('EXAMPLE REVIEWS')
    ex_eng = df[pure_english & (word_counts>10)].head(1)
    if not ex_eng.empty:
        write('[Clean English]')
        write(ex_eng.iloc[0]['review_text'])
    else: write('No good clean English found.')
    ex_mix = df[code_mixed & (word_counts>7)].head(1)
    if not ex_mix.empty:
        write('[Code-Mixed Hindi-English]')
        write(ex_mix.iloc[0]['review_text'])
    else: write('No code-mixed example found.')
    ex_short = df[short_mask].head(1)
    if not ex_short.empty:
        write('[Very Short Review]')
        write(ex_short.iloc[0]['review_text'])
    else: write('No very short review found.')

    section('NOTES')
    write('- Empty text or rating = missing data, should drop')
    write('- Very short (<5 words) rarely useful for sentiment/aspect learning')
    write('- Duplicates indicate possible spam or repetitive sampling')
    write('- Code-mixed = at least one common Hindi token in review_text')
    write('- Pure English = only ascii letters/numbers/punct and NOT code-mixed')
