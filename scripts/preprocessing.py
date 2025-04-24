import pandas as pd
import json


def clean_reviews(df):
    # Remove reviews with missing or blank text
    df = df[df['text'].notnull() & (df['text'].str.strip() != '')].copy()
    # Remove duplicates
    df = df.drop_duplicates(subset=['review_id'])
    # Lowercase text
    df['text'] = df['text'].str.lower()
    # Text length
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    # Date features
    df['review_year'] = pd.to_datetime(df['date']).dt.year
    df['review_month'] = pd.to_datetime(df['date']).dt.month
    return df

def add_category_flags(df):
    df['is_hotel'] = df['categories'].apply(lambda x: 'hotel' in str(x).lower())
    df['is_spa'] = df['categories'].apply(lambda x: 'spa' in str(x).lower())
    df['is_restaurant'] = df['categories'].apply(lambda x: 'restaurant' in str(x).lower())
    return df

def load_businesses(business_path, target_cities, hospitality_keywords, major_chains):
    """Loads and filters businesses by city and category/chain."""
    businesses = []
    with open(business_path, encoding='utf-8') as f:
        for line in f:
            businesses.append(json.loads(line))
    business_df = pd.DataFrame(businesses)
    city_filtered = business_df[business_df['city'].isin(target_cities)].copy()
    
    def is_hospitality(row):
        cat_str = str(row['categories']) if pd.notnull(row['categories']) else ''
        name = str(row['name'])
        return (
            any(kw in cat_str for kw in hospitality_keywords) or
            any(chain in name for chain in major_chains)
        )
    hospitality_filtered = city_filtered[city_filtered.apply(is_hospitality, axis=1)].copy()
    return hospitality_filtered

def flag_major_chains(df, major_chains):
    """Adds a boolean column if the business is a major chain."""
    df['is_chain'] = df['name'].apply(lambda n: any(chain in str(n) for chain in major_chains))
    return df

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
