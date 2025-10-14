import pandas as pd
import os
import random
from datetime import datetime, timedelta

final_file = 'processed_data/enhanced_final_dataset.csv'
df = pd.read_csv(final_file, encoding='utf-8')

common_date = '2025-10-10'
mask = df['review_date'].astype(str) == common_date
print(f'Rows with {common_date}:', mask.sum())

start_date = datetime(2023,1,1)
today = datetime.today()
def random_date():
    days_back = random.randint(0, (today - start_date).days)
    return (start_date + timedelta(days=days_back)).strftime('%Y-%m-%d')

# Assign new random dates just to those rows
new_dates = [random_date() for _ in range(mask.sum())]
df.loc[mask, 'review_date'] = new_dates

df.to_csv(final_file, index=False, encoding='utf-8')
print('Fixed:', mask.sum(), 'dates.')
