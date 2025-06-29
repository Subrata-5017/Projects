# main.py (For local console testing and initial embedding generation)
from src.data_loader import load_and_clean_data
from src.features import combine_features
from src.recommender import generate_embeddings, recommend, clean_title

import os
import difflib
import numpy as np
import pandas as pd

# --- Load and prepare data ---
print("🚀 Loading and cleaning data...")
df = load_and_clean_data()

# Combine features for semantic embeddings
df['combined_features'] = combine_features(df)
print(f"✅ Loaded {len(df)} items and generated combined features.")

# --- Load or generate embeddings ---
embedding_path = "embeddings.npy"
embeddings = None

if os.path.exists(embedding_path):
    print("📂 Loading cached embeddings...")
    embeddings = np.load(embedding_path)
    print(f"✅ Embeddings loaded. Shape: {embeddings.shape}, Dtype: {embeddings.dtype}")

    if embeddings.shape[0] != len(df):
        print("⚠️ Warning: Cached embeddings mismatch — regenerating...")
        embeddings = generate_embeddings(df["combined_features"])
        np.save(embedding_path, embeddings)
        print(f"✅ Regenerated embeddings. Shape: {embeddings.shape}")
else:
    print("🧠 No cached embeddings found — generating...")
    embeddings = generate_embeddings(df["combined_features"])
    np.save(embedding_path, embeddings)
    print(f"✅ Embeddings generated. Shape: {embeddings.shape}")

# --- User interaction in console ---
movie = input("\n🎬 Enter a movie or web series title you like: ").strip()
num = input("📈 How many recommendations? (default 10): ").strip()
content_type_input = input("🎭 Filter by type? (movie/web series/both - default both): ").strip().lower()

# Handle number of recommendations
num_results = 10
if num.isdigit():
    num_results = int(num)

# Normalize content type input
content_type_map = {
    "movie": "movie",
    "webseries": "webseries",
    "web_series": "webseries",
    "web series": "webseries",
    "both": None,
    "": None
}
content_type_filter = content_type_map.get(content_type_input, None)
if content_type_filter is None and content_type_input not in content_type_map:
    print(f"⚠️ Invalid content type filter '{content_type_input}'. Using 'both'.")

# --- Recommendation Logic ---
if not movie:
    print("\n❌ Movie title cannot be empty.")
else:
    results, selected_title, selected_type = recommend(
        movie,
        df,
        embeddings,
        num_results=num_results,
        content_type=content_type_filter
    )

    if results is not None and not results.empty:
        print(f"\n--- Top Recommendations for '{selected_title}' ({selected_type}) ---")
        for _, row in results.iterrows():
            genres = row['genres'].replace(' ', ', ') if isinstance(row['genres'], str) else ''
            platform = row['platform'] if row['platform'] else "N/A"
            seasons_info = f" | Seasons: {int(row['seasons'])}" if row['content_type'] == 'webseries' and row['seasons'] > 0 else ""
            print(f"{row['title']}")
    else:
        print(f"\n⚠️ No recommendations found for '{selected_title}' ({selected_type}) with the given criteria. Try another title or broaden your filter.")
