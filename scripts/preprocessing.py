import pandas as pd
import json


def load_reviews_for_businesses(review_path, valid_business_ids):
    """Loads reviews, keeping only those matching the business IDs."""
    reviews = []
    with open(review_path, encoding='utf-8') as f:
        for line in f:
            review = json.loads(line)
            if review['business_id'] in valid_business_ids:
                reviews.append(review)
    return pd.DataFrame(reviews)

def merge_reviews_businesses(review_df, business_df):
    """Merges review and business data for full context."""
    return review_df.merge(business_df, on='business_id', suffixes=('_review', '_biz'))

def sample_reviews_per_city(df, city_col='city', n_per_city=5000, random_state=42):
    """Returns a DataFrame with up to n reviews per city."""
    sampled = []
    for city in df[city_col].unique():
        city_df = df[df[city_col] == city]
        n = min(n_per_city, len(city_df))
        if n > 0:
            sampled.append(city_df.sample(n=n, random_state=random_state))
    return pd.concat(sampled, ignore_index=True)

def clean_reviews(df):
    # Remove reviews with missing or blank text
    df = df[df['text'].notnull() & (df['text'].str.strip() != '')].copy()
    # Remove duplicates by review_id
    df = df.drop_duplicates(subset=['review_id'])
    # Lowercase and strip text
    df['text'] = df['text'].str.lower().str.strip()
    # Text length (words)
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    # Remove reviews with abnormally short text (optional, e.g., <5 words)
    df = df[df['text_length'] >= 5]
    # Date features
    df['review_year'] = pd.to_datetime(df['date']).dt.year
    df['review_month'] = pd.to_datetime(df['date']).dt.month
    return df

def add_chain_flags(df):
    chains = ['marriott', 'hilton', 'hyatt', 'sheraton', 'westin', 'doubletree', 'holiday inn']
    for chain in chains:
        flag = f'is_{chain.replace(" ", "_")}'
        df[flag] = df['name'].str.contains(chain, case=False, na=False)
    return df

def add_category_flags(df):
    # If 'categories' column exists, create type flags
    if 'categories' in df.columns:
        df['is_hotel'] = df['categories'].str.contains('hotel', case=False, na=False)
        df['is_spa'] = df['categories'].str.contains('spa', case=False, na=False)
        df['is_restaurant'] = df['categories'].str.contains('restaurant', case=False, na=False)
    return df

def get_chain(row):
    for chain in ['marriott', 'hilton', 'hyatt', 'sheraton', 'westin', 'doubletree', 'holiday inn']:
        if row.get(f'is_{chain}', False):
            return chain.title()
    return 'Other'


