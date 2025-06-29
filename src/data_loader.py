# src/data_loader.py
import pandas as pd
import ast
import re
import numpy as np

def parse_names(text):
    """
    Parses a string representation of a list of dictionaries (e.g., for 'cast', 'keywords', 'genres')
    and extracts the 'name' values, joining them into a single string.
    Handles various parsing errors robustly.
    """
    try:
        if pd.isna(text) or not isinstance(text, str):
            return ''
        # Safely evaluate string literal which is a list of dicts
        list_of_dicts = ast.literal_eval(text)
        if not isinstance(list_of_dicts, list): # Ensure it's a list
            return ''
        return ' '.join([d.get('name', '') for d in list_of_dicts if isinstance(d, dict) and 'name' in d])
    except (ValueError, SyntaxError, TypeError):
        # Catch errors from ast.literal_eval, or if text is not a string
        return ''

def extract_director(text):
    """
    Parses a string representation of a list of dictionaries (for 'crew')
    and extracts the 'name' of the person with 'job': 'Director'.
    Handles various parsing errors robustly.
    """
    try:
        if pd.isna(text) or not isinstance(text, str):
            return ''
        crew_list = ast.literal_eval(text)
        if not isinstance(crew_list, list): # Ensure it's a list
            return ''
        for member in crew_list:
            if isinstance(member, dict) and member.get('job') == 'Director':
                return member.get('name', '')
        return ''
    except (ValueError, SyntaxError, TypeError):
        # Catch errors from ast.literal_eval, or if text is not a string
        return ''

def clean_text_simple(text):
    """
    Basic cleaning for text fields, used within data_loader functions to standardize input.
    Removes non-alphanumeric characters (except spaces), converts to lowercase,
    and collapses multiple spaces.
    """
    if isinstance(text, str):
        # Remove non-alphanumeric characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Replace multiple spaces with a single space and strip leading/trailing spaces
        return re.sub(r'\s+', ' ', text).strip()
    return ''

def load_and_clean_data():
    """
    Loads raw movie and web series data, processes, cleans, and combines them
    into a single DataFrame ready for feature engineering.
    Includes optional dataset reduction for deployment.
    """
    # Load movie datasets
    try:
        movie_metadata = pd.read_csv('data/movies_metadata.csv', low_memory=False)
        movie_credits = pd.read_csv('data/credits_small.csv')
        movie_keywords = pd.read_csv('data/keywords.csv')
        print("✅ Movie raw data loaded.")
    except FileNotFoundError as e:
        print(f"ERROR: Missing movie data file: {e}. Make sure 'data/' directory and CSVs exist.")
        raise

    # Process movie data
    movie_df = process_movie_data(movie_metadata, movie_credits, movie_keywords)
    print(f"Processed {len(movie_df)} movie entries.")

    # Load and process web series data
    try:
        web_series = pd.read_csv('data/web_series.csv')
        web_df = process_web_data(web_series)
        print(f"Processed {len(web_df)} web series entries.")
    except FileNotFoundError:
        print("WARNING: 'data/web_series.csv' not found. Skipping web series data.")
        web_df = pd.DataFrame() # Create an empty DataFrame to allow concatenation

    # Combine datasets
    df = pd.concat([movie_df, web_df], ignore_index=True)
    print(f"Combined dataset size: {len(df)} items.")

    # Clean and standardize the 'seasons' column
    if 'seasons' in df.columns:
        df['seasons'] = df['seasons'].astype(str).str.extract(r'(\d+)').fillna('0')
        df['seasons'] = pd.to_numeric(df['seasons'], errors='coerce').fillna(0).astype(int)
    else:
        df['seasons'] = 0 # Ensure 'seasons' column exists

    # Normalize ratings and popularity across both datasets
    df = normalize_ratings(df)
   
    # Final check: Ensure all columns expected by features.py and recommender.py exist
    expected_text_cols = ['keywords', 'cast', 'genres', 'director', 'overview', 'content_rating', 'platform', 'content_type', 'title', 'release_year'] # Added release_year here
    for col in expected_text_cols:
        if col not in df.columns:
            df[col] = ''
        else:
            df[col] = df[col].fillna('').astype(str).apply(clean_text_simple)

    expected_numeric_cols = ['vote_average', 'popularity', 'seasons']
    for col in expected_numeric_cols:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Drop duplicates one final time on combined features if desired
    # This can help remove exact duplicate entries that might have slipped through
    df.drop_duplicates(subset=['title', 'release_year', 'content_type'], inplace=True)

    df['content_type'] = df['content_type'].str.strip().str.lower().replace({
    'web series': 'webseries',
    'web_series': 'webseries',
    'tv show': 'webseries',
    'series': 'webseries',
    'film': 'movie',
    'tv': 'webseries'
    })
    print("✅ Data loading and cleaning complete.")
    
    return df

def normalize_ratings(df):
    """
    Normalizes 'vote_average' and 'popularity' across the combined dataset.
    Standardizes IMDB ratings for web series and scales popularity to 0-100.
    """
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0)
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)

    if 'R Rating' in df.columns:
        df['R Rating'] = pd.to_numeric(df['R Rating'], errors='coerce').fillna(0)
        df['vote_average'] = np.where(
            (df['content_type'] == 'web_series') & (df['R Rating'] > 0),
            df['R Rating'],
            df['vote_average']
        )
        df.drop(columns=['R Rating'], inplace=True, errors='ignore')

    if df['popularity'].max() > 0:
        min_pop = df['popularity'].min()
        max_pop = df['popularity'].max()
        if max_pop == min_pop:
            df['popularity'] = 0
        else:
            df['popularity'] = (df['popularity'] - min_pop) / (max_pop - min_pop) * 100
    else:
        df['popularity'] = 0

    return df

def process_movie_data(metadata, credits, keywords):
    """
    Processes the raw movie metadata, credits, and keywords DataFrames.
    Merges them, parses complex fields, cleans text, and drops unneeded columns.
    """
    # Fix ID columns: convert to numeric, coerce errors, and drop NaNs
    metadata['id'] = pd.to_numeric(metadata['id'], errors='coerce')
    credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
    keywords['id'] = pd.to_numeric(keywords['id'], errors='coerce')

    metadata.dropna(subset=['id'], inplace=True)
    credits.dropna(subset=['id'], inplace=True)
    keywords.dropna(subset=['id'], inplace=True)

    # Ensure IDs are integers for merging
    metadata['id'] = metadata['id'].astype(int)
    credits['id'] = credits['id'].astype(int)
    keywords['id'] = keywords['id'].astype(int)

    # Merge datasets
    df = metadata.merge(credits, on='id', how='left')
    df = df.merge(keywords, on='id', how='left')

    # Apply parsing functions to convert stringified lists/dicts to clean strings
    df['keywords'] = df['keywords'].apply(parse_names)
    df['cast'] = df['cast'].apply(parse_names)
    df['crew'] = df['crew'].apply(extract_director)
    df['genres'] = df['genres'].apply(parse_names)

    # Extract release_year from release_date
    if 'release_date' in df.columns:
        df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
    else:
        df['release_year'] = 0 # Default if release_date is missing

    # Drop the original 'release_date' column as it's no longer needed
    df.drop(columns=['release_date'], inplace=True, errors='ignore')

    # Fill NaNs and ensure string type for text fields
    df['overview'] = df['overview'].fillna('').astype(str)
    df['title'] = df['title'].fillna('').astype(str)
    df['platform'] = '' # Movies metadata typically doesn't have a 'platform', initialize as empty

    # Rename 'crew' column to 'director' for consistency
    df.rename(columns={'crew': 'director'}, inplace=True)
    df['content_type'] = 'movie'

    # Ensure essential numeric columns exist and set defaults
    for col in ['popularity', 'vote_count', 'vote_average', 'seasons', 'content_rating']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Apply general text cleaning to relevant fields for consistency
    text_cols_to_clean = ['genres', 'keywords', 'cast', 'director', 'overview', 'title', 'content_rating', 'platform']
    for col in text_cols_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(clean_text_simple)
        else:
            df[col] = ''

    # Drop other columns not needed further to save memory
    columns_to_drop = [
        'adult', 'belongs_to_collection', 'budget', 'homepage', 'imdb_id', 'original_language',
        'original_title', 'poster_path', 'production_companies', 'production_countries',
        'revenue', 'runtime', 'spoken_languages', 'status', 'tagline', 'video', 'id'
    ]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')

    # Drop duplicates within the movie data itself
    df.drop_duplicates(subset=['title', 'release_year'], inplace=True)

    return df

def process_web_data(web_series):
    """
    Processes the raw web series DataFrame, standardizing column names
    and ensuring consistent data types and defaults, and dropping original year column.
    """
    if web_series.empty:
        # Return an empty DataFrame with all expected columns if web_series is empty
        return pd.DataFrame(columns=[
            'title', 'release_year', 'content_rating', 'vote_average', 'genres',
            'overview', 'seasons', 'platform', 'content_type', 'keywords', 'cast',
            'director', 'vote_count', 'popularity'
        ])

    # CRITICAL CHANGE: Create df by copying relevant columns from web_series AND renaming them directly.
    # This ensures index alignment and correct length.
    df = web_series[[
        'Series Title', 'Year Released', 'Content Rating', 'IMDB Rating',
        'Genre', 'Description', 'No of Seasons', 'Streaming Platform'
    ]].copy()

    # Rename columns to match the movie DataFrame structure
    df.rename(columns={
        'Series Title': 'title',
        'Year Released': 'release_year', # This column now holds the year string
        'Content Rating': 'content_rating',
        'IMDB Rating': 'vote_average',
        'Genre': 'genres',
        'Description': 'overview',
        'No of Seasons': 'seasons',
        'Streaming Platform': 'platform'
    }, inplace=True)

    # Now, fill NaNs on the *newly renamed columns within df*
    df['title'] = df['title'].fillna('')
    df['release_year'] = df['release_year'].fillna('') # Year is kept as string for consistency
    df['content_rating'] = df['content_rating'].fillna('')
    df['vote_average'] = df['vote_average'].fillna(0)
    df['genres'] = df['genres'].fillna('')
    df['overview'] = df['overview'].fillna('')
    df['seasons'] = df['seasons'].fillna(0) # Will be processed by load_and_clean_data for int conversion
    df['platform'] = df['platform'].fillna('')

    df['content_type'] = 'webseries' # Assign constant type here (using 'webseries' to match the cleaned version)

    # Create missing columns with default values, ensuring consistency
    df['keywords'] = ''
    df['cast'] = ''
    df['director'] = ''
    df['vote_count'] = 0 # Default for web series if not available in source
    df['popularity'] = df['vote_average'] * 10 # Simple popularity for web series

    # Apply general text cleaning to relevant fields
    text_cols = ['genres', 'overview', 'content_rating', 'platform', 'title', 'release_year']
    for col in text_cols:
        df[col] = df[col].apply(clean_text_simple)

    # Drop duplicates for web series
    df.drop_duplicates(subset=['title', 'release_year'], inplace=True)

    return df