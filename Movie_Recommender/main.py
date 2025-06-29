# main.py (For local console testing and initial embedding generation)
from src.data_loader import load_and_clean_data
from src.features import combine_features
from src.recommender import generate_embeddings, recommend, clean_title # clean_title is imported from recommender

import os
import difflib # Still useful for initial user input suggestions in console
import numpy as np
import pandas as pd

# Load and prepare data
print("🚀 Loading and cleaning data...")
df = load_and_clean_data()
# This is crucial: combine features BEFORE generating embeddings
# This combined string is what the SentenceTransformer model will encode
df['combined_features'] = combine_features(df)
print(f"Loaded {len(df)} items and generated combined features.")

# Model caching setup for embeddings.npy
embedding_path = "embeddings.npy"
embeddings = None # Initialize embeddings variable

# Load or generate embeddings
if os.path.exists(embedding_path):
    print("📂 Loading cached embeddings...")
    # NumPy automatically loads the correct dtype (float16) if saved as such
    embeddings = np.load(embedding_path)
    print(f"Embeddings loaded. Shape: {embeddings.shape}, Dtype: {embeddings.dtype}")
    # Basic check for consistency: Ensure loaded embeddings match DataFrame size
    if embeddings.shape[0] != len(df):
        print("⚠️ Warning: Loaded embeddings do not match current DataFrame size. Regenerating embeddings.")
        embeddings = generate_embeddings(df["combined_features"])
        np.save(embedding_path, embeddings)
        print("✅ Embeddings regenerated and saved.")
        print(f"Generated embeddings. Shape: {embeddings.shape}, Dtype: {embeddings.dtype}")
else:
    print("🧠 Cached embeddings not found. Generating new embeddings...")
    # Pass the rich 'combined_features' to generate_embeddings for better semantic understanding
    embeddings = generate_embeddings(df["combined_features"])
    np.save(embedding_path, embeddings)
    print("✅ Embeddings generated and saved.")
    print(f"Generated embeddings. Shape: {embeddings.shape}, Dtype: {embeddings.dtype}")


# --- Console User Interaction ---
movie = input("\n🎬 Enter a movie or web series title you like: ").strip()
num = input("📈 How many recommendations? (default 10): ").strip()
content_type_input = input("🎭 Filter by type? (movie/web_series/both - default both): ").strip().lower()

num_results = 10
if num.isdigit():
    num_results = int(num)

content_type_filter = None
if content_type_input in ['movie', 'web_series']:
    content_type_filter = content_type_input
elif content_type_input == 'both' or not content_type_input:
    content_type_filter = None # No filter
else:
    print(f"Invalid content type filter '{content_type_input}'. Using 'both'.")


if not movie:
    print("\n❌ Movie title cannot be empty.")
else:
    # The `recommend` function now handles fuzzy matching and returns the actual matched title/type.
    results, selected_title, selected_type = recommend(
        movie,          # User's raw input title
        df,             # DataFrame with all movie/series data
        embeddings,     # Pre-computed embeddings
        num_results=num_results,
        content_type=content_type_filter
    )

    if results is not None and not results.empty:
        print(f"\n--- Top Recommendations for '{selected_title}' ({selected_type}) ---")
        for i, row in results.iterrows():
            genres = row['genres'].replace(' ', ', ') if isinstance(row['genres'], str) else ', '.join(row['genres'])
            platform = row['platform'] if row['platform'] else "N/A"
            seasons_info = f" | Seasons: {int(row['seasons'])}" if row['content_type'] == 'web_series' and row['seasons'] > 0 else ""
            print(f"{row['title']}")
    else:
        print(f"\n⚠️ No recommendations found for '{selected_title}' ({selected_type}) with the given criteria. Try another title or broaden your filter.")