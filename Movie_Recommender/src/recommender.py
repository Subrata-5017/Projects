# src/recommender.py
import numpy as np
import pandas as pd
import re
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model once when the module is imported
print("🧠 Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ SentenceTransformer model loaded.")


def clean_title(title):
    """
    Standardizes a title for consistent matching within the DataFrame and user input.
    Removes leading/trailing articles, non-alphanumeric, and extra spaces.
    """
    if not isinstance(title, str):
        return ''
    title = title.lower()
    title = re.sub(r'^(the |a |an )', '', title)
    title = re.sub(r'[^a-z0-9 ]', '', title)
    return title.strip()

def generate_embeddings(descriptions):
    """
    Generates embeddings from a list of textual descriptions using SentenceTransformer.
    Embeddings are converted to float16 to reduce file size significantly for deployment.
    """
    if not isinstance(descriptions, pd.Series) and not isinstance(descriptions, list):
        print("Warning: descriptions input is not a pandas Series or list. Attempting conversion.")
        descriptions = pd.Series(descriptions)

    valid_descriptions = descriptions.dropna().astype(str).tolist()
    if not valid_descriptions:
        print("No valid descriptions found to generate embeddings.")
        return np.array([], dtype=np.float16).reshape(0, model.get_sentence_embedding_dimension())

    print(f"Encoding {len(valid_descriptions)} items into embeddings...")
    embeddings = model.encode(valid_descriptions, show_progress_bar=True, batch_size=32)

    full_embeddings = np.zeros((len(descriptions), model.get_sentence_embedding_dimension()), dtype=np.float32)
    valid_indices = descriptions.dropna().index
    full_embeddings[valid_indices] = embeddings

    print("Converting embeddings to float16 for storage efficiency...")
    return full_embeddings.astype(np.float16)

def recommend(query_title, df, embeddings, num_results=10, content_type=None):
    """
    Generates content-based recommendations based on cosine similarity and custom boosts.
    Returns:
        - pd.DataFrame: Top recommendations.
        - str: The actual title of the matched query item in the dataset.
        - str: The content type of the matched query item.
    """
    df_copy = df.copy()

    if 'cleaned_title' not in df_copy.columns:
        df_copy['cleaned_title'] = df_copy['title'].apply(clean_title)

    cleaned_query = clean_title(query_title)
    all_cleaned_titles = df_copy['cleaned_title'].tolist()
    close_matches = difflib.get_close_matches(cleaned_query, all_cleaned_titles, n=1, cutoff=0.4)

    if not close_matches:
        print(f"DEBUG(recommender): No close match found for '{query_title}'.")
        return pd.DataFrame(), query_title, "N/A"

    best_match_cleaned_title = close_matches[0]
    idx_candidates = df_copy[df_copy['cleaned_title'] == best_match_cleaned_title]
    if idx_candidates.empty:
        print(f"DEBUG(recommender): Matched title '{best_match_cleaned_title}' found by difflib but not in DataFrame indices.")
        return pd.DataFrame(), query_title, "N/A"

    idx = idx_candidates.nlargest(1, 'popularity').index[0]

    original_matched_title = df_copy.loc[idx, 'title']
    original_matched_type = df_copy.loc[idx, 'content_type']

    print(f"\n🔍 Input '{query_title}' matched to: '{original_matched_title}' ({original_matched_type})")
    print(f"DEBUG(recommender): Request content_type filter: {content_type}") # This is the incoming filter

    query_vec = embeddings[idx].reshape(1, -1)
    sim_scores = cosine_similarity(query_vec, embeddings)[0]
    df_copy['similarity'] = sim_scores

    def to_set(text):
        return set(str(text).lower().split())

    query_genres = to_set(df_copy.at[idx, 'genres'])
    query_keywords = to_set(df_copy.at[idx, 'keywords'])
    query_type = df_copy.at[idx, 'content_type']

    def compute_overlap_boost(row_set, query_set, power=1.0):
        if not query_set:
            return 1.0
        overlap = len(row_set.intersection(query_set))
        union = len(row_set.union(query_set))
        base = 1.0 + (overlap / union) if union else 1.0
        return base ** power

    df_copy['genre_boost'] = df_copy['genres'].apply(lambda g: compute_overlap_boost(to_set(g), query_genres, power=1.5))
    df_copy['keyword_boost'] = df_copy['keywords'].apply(lambda k: compute_overlap_boost(to_set(k), query_keywords, power=1.2))
    df_copy['type_boost'] = df_copy['content_type'].apply(lambda t: 1.2 if t == query_type else 1.0)
    df_copy['rating_boost'] = 1 + df_copy['vote_average'] / 10 * 0.4
    df_copy['popularity_boost'] = 1 + df_copy['popularity'] / 100 * 0.3
    # Increased seasons_boost to give more weight to web series with seasons
    df_copy['seasons_boost'] = df_copy['seasons'].apply(lambda s: 1.0 + (0.2 if original_matched_type == 'web_series' and s > 0 else 0) ) # Increased from 0.1 to 0.2

    df_copy['final_score'] = (
        df_copy['similarity'] *
        df_copy['genre_boost'] *
        df_copy['keyword_boost'] *
        df_copy['type_boost'] *
        df_copy['rating_boost'] *
        df_copy['popularity_boost'] *
        df_copy['seasons_boost']
    )

    df_recs = df_copy[df_copy.index != idx].copy()

    # print(f"DEBUG(recommender): Candidates BEFORE type filter: {len(df_recs)}")
    # print(f"DEBUG(recommender): Content types in candidates before filter: {df_recs['content_type'].unique().tolist()}")

    if content_type and content_type in ['movie', 'web_series']:
        initial_count = len(df_recs)
        df_recs = df_recs[df_recs['content_type'] == content_type]
        print(f"DEBUG(recommender): Type filter '{content_type}' applied. Candidates remaining: {len(df_recs)} (from {initial_count})")
        if df_recs.empty:
            print(f"DEBUG(recommender): DataFrame is empty AFTER content_type filter '{content_type}'. This is why no recommendations are returned.")
            return pd.DataFrame(), original_matched_title, original_matched_type

    # --- NEW DEBUGGING PRINTS ---
    if not df_recs.empty:
        # print(f"DEBUG(recommender): Max final_score for filtered candidates: {df_recs['final_score'].max():.4f}")
        # print(f"DEBUG(recommender): Min final_score for filtered candidates: {df_recs['final_score'].min():.4f}")
        # print(f"DEBUG(recommender): Mean final_score for filtered candidates: {df_recs['final_score'].mean():.4f}")

        # Calculate star_score for all remaining candidates for inspection
        max_score = df_recs['final_score'].max()
        min_score = df_recs['final_score'].min()
        if max_score == min_score:
            df_recs['star_score'] = 0
        else:
            df_recs['star_score'] = 5 * (df_recs['final_score'] - min_score) / (max_score - min_score)

        # print(f"DEBUG(recommender): Top 5 filtered candidates by final_score (before threshold):")
        # print(df_recs[['title', 'content_type', 'final_score', 'star_score']].sort_values('final_score', ascending=False).head(5))

        top_candidates = df_recs[df_recs['star_score'] >= 3].sort_values('final_score', ascending=False)
        # print(f"DEBUG(recommender): Number of candidates meeting star_score >= 3: {len(top_candidates)}")
        if top_candidates.empty:
            # print("DEBUG(recommender): No candidates meet the star_score >= 3 threshold.")
            # If nothing meets threshold, get the best available, even if score is low
            fallback = df_recs.sort_values('final_score', ascending=False).head(num_results)
            if not fallback.empty:
                print(f"DEBUG(recommender): Falling back to top {len(fallback)} candidates regardless of star_score.")
                final_recommendations = fallback
            else:
                return pd.DataFrame(), original_matched_title, original_matched_type
        else:
            # If enough top candidates, use them
            if len(top_candidates) >= num_results:
                final_recommendations = top_candidates.head(num_results)
            else:
                # If some meet the threshold, but not enough, supplement with next best
                needed = num_results - len(top_candidates)
                fallback = df_recs[~df_recs.index.isin(top_candidates.index)].sort_values('final_score', ascending=False).head(needed)
                final_recommendations = pd.concat([top_candidates, fallback]).drop_duplicates(subset=['title']).head(num_results) # Added drop_duplicates after concat if needed
    else:
        print("DEBUG(recommender): df_recs is empty before star_score check.")
        return pd.DataFrame(), original_matched_title, original_matched_type

    final_recommendations = final_recommendations[[
        'title', 'content_type', 'vote_average', 'genres', 'platform', 'popularity', 'seasons'
    ]]

    print("\n🎯 Recommendations successfully generated.")
    return final_recommendations, original_matched_title, original_matched_type