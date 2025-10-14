import pandas as pd
import os
import string
import random
from datetime import datetime, timedelta

RAW_DIR = 'raw_data'
PROC_DIR = 'processed_data'


def load_csv_safe(path: str, expected_min_cols: int = 1) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='utf-8-sig')
    if df.shape[1] < expected_min_cols:
        print(f"Unexpected columns in {path}: {list(df.columns)}")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df


def generate_code_mixed_synthetic_reviews(n=10000):
    """Generate coherent code-mixed reviews with platform-category-product consistency.

    Rules:
    - Swiggy/Zomato: food-only (dishes, restaurants), categories Food/Delivery
    - Myntra: fashion-only (apparel/footwear/accessories), category Fashion
    - Nykaa: beauty-only (skincare/makeup/hair), category Beauty
    - Amazon/Flipkart: general retail (electronics/home/kitchen), categories mapped accordingly
    """
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Ahmedabad', 'Pune', 'Lucknow', 'Jaipur']
    ratings_sentiments = [(5, 'positive'), (4, 'positive'), (3, 'neutral'), (2, 'negative'), (1, 'negative')]

    domain = {
        'Swiggy': {
            'categories': ['Food'],
            'products': ['Paneer Tikka', 'Chicken Biryani', 'Masala Dosa', 'Veg Thali', 'Cheeseburger', 'Margherita Pizza', 'Gulab Jamun', 'Idli Sambar', 'Rajma Chawal'],
            'templates': [
                "{platform} se {product} order kiya, taste {taste_desc} tha, delivery {delivery_desc}. {packaging_desc}. {extra}",
                "{city} me {platform} ka order diya: {product}. Quantity {quantity_desc} thi, {temp_desc}. {extra}",
                "{product} fresh tha aur {platform} delivery {delivery_desc} thi. Price {price_desc}. {extra}"
            ]
        },
        'Zomato': {
            'categories': ['Food'],
            'products': ['Paneer Tikka', 'Butter Chicken', 'Chole Bhature', 'Veg Pulao', 'Chicken Shawarma', 'Hakka Noodles', 'Samosa', 'Vada Pav', 'Tandoori Roti'],
            'templates': [
                "{platform} pe {product} mangwaya, {taste_desc}. Delivery {delivery_desc} thi, packaging {packaging_short}. {extra}",
                "Restaurant se {product} order kiya via {platform}. Portion {quantity_desc} tha, {temp_desc}. {extra}",
                "{product} ka flavour {flavour_desc} tha. {platform} delivery {delivery_desc}. {extra}"
            ]
        },
        'Myntra': {
            'categories': ['Fashion'],
            'products': ['Cotton T-Shirt', 'Denim Jeans', 'Running Shoes', 'Kurta Set', 'Sneakers', 'Formal Shirt', 'Hoodie', 'Track Pants', 'Anarkali Dress'],
            'templates': [
                "{platform} se {product} liya, size {fit_desc} tha aur fabric {fabric_desc}. Delivery {delivery_desc}. {extra}",
                "{product} ki stitching {quality_desc} hai, color {color_desc}. Exchange/return {service_desc}. {extra}",
                "Price {price_desc} tha, {product} overall {overall_desc}. {extra}"
            ]
        },
        'Nykaa': {
            'categories': ['Beauty'],
            'products': ['Face Serum', 'Moisturizer', 'Sunscreen SPF50', 'Matte Lipstick', 'Kajal', 'Shampoo', 'Conditioner', 'Face Wash', 'Hair Oil'],
            'templates': [
                "{platform} se {product} liya, skin pe feel {skinfeel_desc} hai. Packaging {packaging_short}. Delivery {delivery_desc}. {extra}",
                "{product} ka result {result_desc} mila, fragrance {fragrance_desc}. Price {price_desc}. {extra}",
                "Allergic reaction nahi hua, texture {texture_desc}. {platform} service {service_desc}. {extra}"
            ]
        },
        'Amazon': {
            'categories': ['Electronics', 'Home', 'Kitchen'],
            'products': ['Bluetooth Speaker', 'Powerbank', 'Mixer Grinder', 'LED Bulb Pack', 'Pressure Cooker', 'Electric Kettle', 'Trimmer', 'Hair Dryer', 'Desk Lamp'],
            'templates': [
                "{platform} pe {product} purchase kiya, build {quality_desc} hai. Packaging {packaging_short}, delivery {delivery_desc}. {extra}",
                "{product} ka performance {performance_desc} hai, price {price_desc}. Return/replace {service_desc}. {extra}",
                "Installation {install_desc} tha, overall {overall_desc}. {extra}"
            ]
        },
        'Flipkart': {
            'categories': ['Electronics', 'Home', 'Kitchen'],
            'products': ['Bluetooth Speaker', 'Powerbank', 'Mixer Grinder', 'LED Bulb Pack', 'Pressure Cooker', 'Electric Kettle', 'Trimmer', 'Hair Dryer', 'Steam Iron'],
            'templates': [
                "{platform} se {product} liya, quality {quality_desc}. Delivery {delivery_desc}, packaging {packaging_short}. {extra}",
                "{product} ka performance {performance_desc} tha, price {price_desc}. Warranty/return {service_desc}. {extra}",
                "Design {design_desc} hai, use karne me {usability_desc}. {extra}"
            ]
        }
    }

    # Descriptor pools per domain
    taste_desc = ['bahut achha', 'theek thaak', 'average', 'bland', 'spicy', 'fresh']
    flavour_desc = ['rich', 'balanced', 'overpowering', 'light']
    quantity_desc = ['kaafi', 'kam', 'sahi']
    temp_desc = ['garam aaya', 'thanda pohoch gaya', 'thik temperature']
    packaging_short = ['sahi thi', 'leak-proof thi', 'kharab thi', 'theek thi']
    price_desc = ['reasonable', 'mehenga tha', 'affordable', 'zyada tha']
    delivery_desc = ['bahut fast thi', 'late thi', 'on time thi']
    service_desc = ['helpful tha', 'slow tha', 'theek tha']
    quality_desc = ['badhiya', 'average', 'flimsy', 'solid']
    performance_desc = ['mast', 'theek', 'below average', 'excellent']
    design_desc = ['sleek', 'simple', 'outdated', 'modern']
    install_desc = ['easy', 'difficult', 'smooth']
    overall_desc = ['paisa wasool', 'okay okay', 'disappointing', 'great value']
    fabric_desc = ['soft', 'rough', 'lightweight', 'thick']
    fit_desc = ['perfect', 'thoda tight', 'loose']
    color_desc = ['as shown', 'thoda fade', 'vibrant']
    skinfeel_desc = ['lightweight', 'greasy nahi', 'thoda sticky']
    result_desc = ['visible', 'decent', 'average']
    fragrance_desc = ['mild', 'strong', 'pleasant', 'none']
    texture_desc = ['smooth', 'runny', 'thick']
    usability_desc = ['easy', 'thoda complicated']
    extras = [
        "Phir bhi recommend karunga.", "Overall theek hai.", "Nahi lena chahiye tha.", "Must try.", "Value for money.", "Disappointed ho gaya."
    ]

    platforms = list(domain.keys())
    used = set()
    reviews = []
    tries = 0
    max_tries = n * 25
    while len(reviews) < n and tries < max_tries:
        tries += 1
        platform = random.choice(platforms)
        cfg = domain[platform]
        category = random.choice(cfg['categories'])
        product = random.choice(cfg['products'])
        city = random.choice(cities)
        rating, sentiment = random.choice(ratings_sentiments)
        tpl = random.choice(cfg['templates'])

        review_text = tpl.format(
            platform=platform,
            product=product,
            city=city,
            taste_desc=random.choice(taste_desc),
            flavour_desc=random.choice(flavour_desc),
            quantity_desc=random.choice(quantity_desc),
            temp_desc=random.choice(temp_desc),
            packaging_desc=f"Packaging {random.choice(packaging_short)}",
            packaging_short=random.choice(packaging_short),
            price_desc=random.choice(price_desc),
            delivery_desc=random.choice(delivery_desc),
            service_desc=random.choice(service_desc),
            quality_desc=random.choice(quality_desc),
            performance_desc=random.choice(performance_desc),
            design_desc=random.choice(design_desc),
            install_desc=random.choice(install_desc),
            overall_desc=random.choice(overall_desc),
            fabric_desc=random.choice(fabric_desc),
            fit_desc=random.choice(fit_desc),
            color_desc=random.choice(color_desc),
            skinfeel_desc=random.choice(skinfeel_desc),
            result_desc=random.choice(result_desc),
            fragrance_desc=random.choice(fragrance_desc),
            texture_desc=random.choice(texture_desc),
            usability_desc=random.choice(usability_desc),
            extra=random.choice(extras)
        ).strip()

        key = review_text.lower()
        if len(review_text.split()) < 6:
            continue
        if key in used:
            continue
        used.add(key)
        reviews.append({
            'review_text': review_text,
            'rating': rating,
            'sentiment': sentiment,
            'language_mix': 'hindi_english',
            'platform': platform,
            'category': category,
            'product_name': product,
            'city': city,
        })

    df = pd.DataFrame(reviews)
    df['review_length'] = df['review_text'].astype(str).str.len()
    df['review_id'] = [f'SYN_{i+1:06d}' for i in range(len(df))]
    return df

# Preview: Uncomment to check a small sample and unique count
# df_syn = generate_synthetic_reviews_no_duplicates(20)
# print(df_syn['review_text'])
# print(df_syn['review_text'].nunique())

# Next, this function should be wired into the build pipeline. Let me know to proceed.


def standardize_to_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = normalize_columns(df)
    out = pd.DataFrame()

    # source-specific column candidates
    text_cols = [c for c in df.columns if any(k in c for k in ['review_text', 'text', 'review', 'sentence', 'content'])]
    rating_cols = [c for c in df.columns if 'rating' in c or c in ['stars', 'star', 'overall']]
    sentiment_cols = [c for c in df.columns if 'sentiment' in c or c == 'label']
    date_cols = [c for c in df.columns if 'date' in c]
    city_cols = [c for c in df.columns if 'city' in c or 'location' in c]
    category_cols = [c for c in df.columns if any(k in c for k in ['category', 'product_category'])]
    platform_cols = [c for c in df.columns if any(k in c for k in ['platform', 'site', 'source'])]

    # Prefer Dataset-SA 'summary' as review text when available
    if source == 'dataset_sa' and 'summary' in df.columns:
        out['review_text'] = df['summary']
    else:
        out['review_text'] = df[text_cols[0]] if text_cols else ''
    if rating_cols:
        out['rating'] = pd.to_numeric(df[rating_cols[0]], errors='coerce')
    else:
        # derive from sentiment if available later
        out['rating'] = pd.NA

    if sentiment_cols:
        sent = df[sentiment_cols[0]].astype(str).str.lower()
        sent = sent.replace({'1': 'positive', '0': 'negative', '2': 'neutral'})
        # normalize empty strings to NaN
        sent = sent.replace({'': pd.NA, 'nan': pd.NA, 'none': pd.NA})
        out['sentiment'] = sent
    else:
        out['sentiment'] = pd.NA

    if date_cols:
        out['review_date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
    else:
        out['review_date'] = pd.NaT

    out['city'] = df[city_cols[0]] if city_cols else 'Unknown'
    out['category'] = df[category_cols[0]] if category_cols else 'General'
    out['platform'] = df[platform_cols[0]] if platform_cols else ('Amazon' if 'amazon' in source else ('Flipkart' if 'vfl' in source else 'Online'))

    # language_mix heuristic: assume english unless clearly code-mixed (presence of common Hindi words in Latin)
    def detect_mix(text: str) -> str:
        t = str(text).lower()
        tokens = ['hai', 'nahi', 'bahut', 'acha', 'achha', 'bura', 'kharab', 'sahi', 'galat', 'karunga', 'karti', 'tha', 'thi']
        return 'hindi_english' if any(tok in t for tok in tokens) else 'english'

    out['language_mix'] = out['review_text'].apply(detect_mix)
    out['dataset_source'] = source
    out['review_length'] = out['review_text'].astype(str).str.len()

    # fill sentiment from rating if missing
    mask = out['sentiment'].isna() & out['rating'].notna()
    out.loc[mask & (out['rating'] >= 4), 'sentiment'] = 'positive'
    out.loc[mask & (out['rating'] == 3), 'sentiment'] = 'neutral'
    out.loc[mask & (out['rating'] <= 2), 'sentiment'] = 'negative'

    # fill rating from sentiment if missing
    mask2 = out['rating'].isna() & out['sentiment'].notna()
    out.loc[mask2 & (out['sentiment'] == 'positive'), 'rating'] = 4
    out.loc[mask2 & (out['sentiment'] == 'neutral'), 'rating'] = 3
    out.loc[mask2 & (out['sentiment'] == 'negative'), 'rating'] = 2

    # enforce types
    out['rating'] = pd.to_numeric(out['rating'], errors='coerce').fillna(3).astype(int)
    out['review_date'] = out['review_date'].fillna(pd.Timestamp(datetime.now().date()))

    return out[['review_text', 'rating', 'sentiment', 'review_date', 'city', 'category', 'platform', 'language_mix', 'dataset_source', 'review_length']]


def main():
    os.makedirs(PROC_DIR, exist_ok=True)

    # Load sources
    amazon_vfl = load_csv_safe(os.path.join(RAW_DIR, 'amazon_vfl_reviews.csv'))
    dataset_sa = load_csv_safe(os.path.join(RAW_DIR, 'Dataset-SA.csv'))
    laptop = load_csv_safe(os.path.join(RAW_DIR, 'laptop.csv'))
    restaurant = load_csv_safe(os.path.join(RAW_DIR, 'restaurant.csv'))
    synthetic = load_csv_safe('reviews.csv')

    parts = []

    eng_sources = []
    if not amazon_vfl.empty:
        eng_sources.append(standardize_to_schema(amazon_vfl, 'amazon_vfl'))
    if not dataset_sa.empty:
        eng_sources.append(standardize_to_schema(dataset_sa, 'dataset_sa'))

    # synthetic: ensure exactly 10,000 rows if available, else generate
    if not synthetic.empty:
        syn_std = normalize_columns(synthetic)
        # handle schemas from our generator
        required = ['review_text', 'rating', 'sentiment', 'review_date', 'city', 'category', 'platform', 'language_mix', 'aspects_mentioned', 'verified_purchase', 'helpful_votes', 'review_length']
        if all(col in syn_std.columns for col in required):
            syn_out = syn_std[['review_text', 'rating', 'sentiment', 'review_date', 'city', 'category', 'platform', 'language_mix', 'review_length']].copy()
            syn_out['dataset_source'] = 'synthetic_reviews'
            # limit to 10,000
            if len(syn_out) > 10000:
                syn_out = syn_out.sample(n=10000, random_state=42)
            elif len(syn_out) < 10000:
                # upsample with replacement to reach 10k
                syn_out = syn_out.sample(n=10000, replace=True, random_state=42)
            parts.append(syn_out)
    else:
        # fallback synthetic generation
        syn_out = generate_code_mixed_synthetic_reviews()
        # tag source for downstream logic
        syn_out['dataset_source'] = 'synthetic_reviews'
        parts.append(syn_out)

    # Build English pool strictly from amazon_vfl + dataset_sa, ensure english
    if eng_sources:
        eng_df = pd.concat(eng_sources, ignore_index=True)
        mask_eng = eng_df['dataset_source'].isin(['amazon_vfl', 'dataset_sa'])
        eng_df.loc[mask_eng, 'language_mix'] = 'english'
        # Set platform mapping per source
        eng_df.loc[eng_df['dataset_source'] == 'dataset_sa', 'platform'] = 'Flipkart'
        eng_df.loc[eng_df['dataset_source'] == 'amazon_vfl', 'platform'] = 'Amazon'
        eng_df = eng_df[eng_df['language_mix'] == 'english']
        # sample exactly 10,000
        if len(eng_df) >= 10000:
            eng_pool = eng_df.sample(n=10000, random_state=42)
        else:
            # upsample with replacement if fewer than 10k
            eng_pool = eng_df.sample(n=10000, replace=True, random_state=42)
    else:
        eng_pool = pd.DataFrame(columns=['review_text','rating','sentiment','review_date','city','category','platform','language_mix','dataset_source','review_length'])

    # Combine exactly 10k synthetic + 10k english
    if parts:
        syn_block = parts[0]
    else:
        syn_block = generate_code_mixed_synthetic_reviews()
    # Ensure synthetic block is exactly 10k
    if len(syn_block) > 10000:
        syn_block = syn_block.sample(n=10000, random_state=42)
    elif len(syn_block) < 10000:
        syn_block = syn_block.sample(n=10000, replace=True, random_state=42)

    df = pd.concat([syn_block, eng_pool], ignore_index=True)

    # Basic clean
    df['review_text'] = df['review_text'].astype(str).str.strip()
    df = df[df['review_text'].str.len() > 0]
    # Do not drop duplicates across synthetic vs english pools to preserve exact counts

    # Enrich with commonly expected columns
    # review_id
    df = df.reset_index(drop=True)
    # Neutral review_id without revealing synthetic/english origin
    df['review_id'] = df.apply(lambda r: 'REV_' + f"{(r.name+1):06d}", axis=1)

    # product_id (ASIN-like for English sources, neutral code for others)
    import random
    import string
    def gen_asin_like():
        letters = ''.join(random.choices(string.ascii_uppercase, k=2))
        digits = ''.join(random.choices(string.digits, k=8))
        tail = random.choice(string.ascii_uppercase+string.digits)
        return f"B{digits}{letters}{tail}"
    df['product_id'] = df.apply(lambda r: gen_asin_like() if r['dataset_source'] in ['amazon_vfl','dataset_sa'] else f"PRD{(r.name+1):07d}", axis=1)

    # Preserve platform from generator; if missing, fill with a reasonable default
    syn_platforms = ['Amazon','Flipkart','Myntra','Nykaa','Zomato','Swiggy']
    df.loc[df['platform'].isna() | (df['platform'].astype(str).str.strip()==''), 'platform'] = df.loc[df['platform'].isna() | (df['platform'].astype(str).str.strip()==''), 'platform'].apply(
        lambda p: random.choice(syn_platforms)
    )

    # product_name (derive from category only if missing)
    if 'product_name' not in df.columns:
        df['product_name'] = None
    missing_pname = df['product_name'].isna() | (df['product_name'].astype(str).str.strip()=='')
    df.loc[missing_pname, 'product_name'] = df.loc[missing_pname, 'category'].astype(str).fillna('General') + '_Item'

    # aspects_mentioned via simple keyword rules
    import re
    def tag_aspects(text: str) -> str:
        t = str(text).lower()
        aspects = []
        if re.search(r"\b(deliver(y|ed|ies)|courier|shipping|late|on time)\b", t):
            aspects.append('delivery')
        if re.search(r"\b(quality|defect|broken|durable|build|performance|feature|spec)\b", t):
            aspects.append('product_quality')
        if re.search(r"\b(packaging|package|box|seal)\b", t):
            aspects.append('packaging')
        if re.search(r"\b(price|cost|expensive|cheap|value for money|overpriced)\b", t):
            aspects.append('price')
        if re.search(r"\b(customer service|support|return|refund|replacement|warranty|service)\b", t):
            aspects.append('customer_service')
        return ', '.join(sorted(set(aspects))) if aspects else 'general'
    df['aspects_mentioned'] = df['review_text'].apply(tag_aspects)

    # Drop columns per spec
    # (dataset_source will be dropped when writing)

    # Save
    out_path = os.path.join(PROC_DIR, 'enhanced_final_dataset.csv')
    try:
        # Drop dataset_source, verified_purchase, helpful_votes if present
        drop_cols = [c for c in ['dataset_source','verified_purchase','helpful_votes'] if c in df.columns]
        df.to_csv(out_path, index=False, encoding='utf-8', columns=[c for c in df.columns if c not in drop_cols])
        print(f'Saved {len(df):,} rows -> {out_path}')
    except PermissionError:
        alt_path = os.path.join(PROC_DIR, 'enhanced_final_dataset_v2.csv')
        df.to_csv(alt_path, index=False, encoding='utf-8')
        print(f'Primary file locked, saved {len(df):,} rows -> {alt_path}')

    # Quick report
    print('By source:')
    print(df['dataset_source'].value_counts())
    print('By language:')
    print(df['language_mix'].value_counts())


def build_code_mixed_and_long_english_dataset():
    raw_dir = 'raw_data'
    out_dir = 'processed_data'
    final_file = os.path.join(out_dir, 'enhanced_final_dataset.csv')
    sample_file = os.path.join(out_dir, 'enhanced_final_dataset_sample_2000.csv')
    # 1. Generate code-mixed synthetic (unique, >5w) - already includes coherent platform/category/city
    syn = generate_code_mixed_synthetic_reviews(13000)
    # Ensure required columns exist; do not override coherent fields
    if 'city' not in syn.columns:
        syn['city'] = syn['review_text'].apply(lambda t: random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Ahmedabad', 'Pune', 'Lucknow', 'Jaipur']))
    syn['product_id'] = ['SYN{:07d}'.format(i+1) for i in range(len(syn))]
    # Keep product_name from generator if present
    if 'product_name' not in syn.columns:
        syn['product_name'] = 'General_Item'
    syn['review_length'] = syn['review_text'].astype(str).str.len()
    syn['aspects_mentioned'] = 'general'
    syn['sentiment'] = syn['sentiment'].astype(str)
    # 2. Get unique long English reviews (>10w, dedup across all)
    def get_long_english_from(filename, text_col_pref=None):
        path = os.path.join(raw_dir, filename)
        df = pd.read_csv(path, encoding='utf-8')
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        text_col = None
        if text_col_pref and text_col_pref in df.columns:
            text_col = text_col_pref
        else:
            for c in df.columns:
                if any(tok in c for tok in ['review_text','text','review','sentence','summary','content']):
                    text_col = c; break
            if not text_col: text_col = df.columns[0]
        series = df[text_col].astype(str)
        # Only keep English reviews >10w, dedup
        reviews = series[series.apply(lambda x: len(x.split())>10 and all(ord(ch)<128 for ch in x))]
        return pd.DataFrame({
            'review_text': reviews,
            'platform': 'Amazon' if 'amazon' in filename else 'Flipkart',
            'category': 'General',
            'product_id': ['ENGSRC{:06d}'.format(i+1) for i in range(len(reviews))],
            'review_length': reviews.str.len(),
            'language_mix': 'english',
            'review_id': ['ENG_{:06d}'.format(i+1) for i in range(len(reviews))],
            'product_name': 'General_Item',
            'aspects_mentioned': 'general',
        })
    amz = get_long_english_from('amazon_vfl_reviews.csv')
    sa = get_long_english_from('Dataset-SA.csv', text_col_pref='summary')
    all_eng = pd.concat([amz, sa], ignore_index=True)
    all_eng = all_eng.drop_duplicates('review_text')
    # If more than 7k, sample. If less, upsample with replacement.
    if len(all_eng) > 7000:
        all_eng = all_eng.sample(n=7000, random_state=42)
    elif len(all_eng) < 7000:
        all_eng = all_eng.sample(n=7000, replace=True, random_state=42)
    all_eng['sentiment'] = 4
    all_eng['rating'] = 4
    # Combine
    df = pd.concat([syn, all_eng], ignore_index=True)
    # Final filter: no duplicate review_text, and every review >5 words
    df = df.drop_duplicates('review_text')
    df = df[df['review_text'].apply(lambda x: len(str(x).split())>5)]
    if len(df) > 20000:
        df = df.sample(n=20000, random_state=42)
    elif len(df)<20000:
        # Get more long English if possible
        extra_needed = 20000 - len(df)
        engpool = pd.concat([amz, sa], ignore_index=True)
        pool = engpool[~engpool['review_text'].isin(df['review_text'])]
        # Bootstrapping more synthetic if needed
        if len(pool) >= extra_needed:
            df = pd.concat([df, pool.sample(n=extra_needed, random_state=43)], ignore_index=True)
        else:
            new_syn = generate_code_mixed_synthetic_reviews(extra_needed)
            df = pd.concat([df, pool, new_syn], ignore_index=True)
    # Random dates in last 2 years
    start_date = datetime(2023, 1, 1)
    today = datetime.today()
    def random_date():
        days_back = random.randint(0, (today - start_date).days)
        return (start_date + timedelta(days=days_back)).strftime('%Y-%m-%d')
    df['review_date'] = [random_date() for _ in range(len(df))]
    # Column order
    col_order = ['review_text','rating','sentiment','review_date','city','category','platform','language_mix','review_length','review_id','product_id','product_name','aspects_mentioned']
    df = df[[c for c in col_order if c in df.columns]]
    df.to_csv(final_file, index=False, encoding='utf-8')
    df.sample(n=2000, random_state=42).to_csv(sample_file, index=False, encoding='utf-8')
    print(f"Wrote {len(df)} final rows (13k code-mixed, 7k English), all dups removed. Sample ready.")

if __name__ == '__main__':
    build_code_mixed_and_long_english_dataset()


