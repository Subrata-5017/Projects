# app.py (For Flask Web Application Deployment)
from flask import Flask, render_template, request, jsonify
from src.data_loader import load_and_clean_data
from src.features import combine_features # CRITICAL: Import combine_features for rich embeddings
from src.recommender import generate_embeddings, recommend, clean_title

import os
import numpy as np
import pandas as pd

app = Flask(__name__)
# For production, it's good to disable pretty printing JSON to save bandwidth
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

# --- GLOBAL DATA LOADING & EMBEDDING GENERATION ---
# This block runs once when the Flask application starts.
print("🚀 Initializing Flask App: Loading data and embeddings...")

# Load and preprocess data
# This creates the DataFrame 'df' with all necessary columns
try:
    df = load_and_clean_data()
    print(f"Loaded {len(df)} items into DataFrame.")
except Exception as e:
    print(f"FATAL ERROR: Failed to load and clean data. Application cannot start. Error: {e}")
    # In a production environment, you might want to log this and exit gracefully
    exit()

# Generate the 'combined_features' column which is the input for embeddings
# This uses the logic from src.features.py and is crucial for relevancy
df['combined_features'] = combine_features(df)
print("Combined features generated for all items.")

# Define the path for cached embeddings
embedding_path = "embeddings.npy"
embeddings = None # Initialize embeddings variable

# Check if pre-computed embeddings exist
if os.path.exists(embedding_path):
    print("📂 Loading cached embeddings from disk...")
    # np.load will automatically detect the dtype (float16) if saved as such
    try:
        embeddings = np.load(embedding_path)
        print(f"Embeddings loaded. Shape: {embeddings.shape}, Dtype: {embeddings.dtype}")
        # Basic consistency check: if DataFrame size changed (e.g., due to filtering), regenerate
        if embeddings.shape[0] != len(df):
            print("⚠️ Warning: Loaded embeddings do not match current DataFrame size. Regenerating embeddings.")
            embeddings = generate_embeddings(df["combined_features"]) # Regenerate with correct data source
            np.save(embedding_path, embeddings)
            print("✅ Embeddings regenerated and saved.")
            print(f"Generated embeddings. Shape: {embeddings.shape}, Dtype: {embeddings.dtype}")
        else:
            print("✅ Cached embeddings are consistent with current data.")
    except Exception as e:
        print(f"❌ Error loading cached embeddings: {e}. Regenerating...")
        embeddings = generate_embeddings(df["combined_features"]) # Regenerate if load fails
        np.save(embedding_path, embeddings)
        print("✅ Embeddings generated and saved after error.")
        print(f"Generated embeddings. Shape: {embeddings.shape}, Dtype: {embeddings.dtype}")
else:
    # If not, generate them and save
    print("🔧 Cached embeddings not found. Generating new embeddings...")
    # IMPORTANT: Generate embeddings from 'combined_features' for richer semantic context
    embeddings = generate_embeddings(df["combined_features"])
    np.save(embedding_path, embeddings)
    print("✅ Embeddings generated and saved.")
    print(f"Generated embeddings. Shape: {embeddings.shape}, Dtype: {embeddings.dtype}")

# Ensure embeddings are not None before starting the app
if embeddings is None or embeddings.shape[0] == 0:
    print("FATAL ERROR: Embeddings could not be loaded or generated. Application cannot start.")
    exit()

# --- FLASK ROUTES ---

@app.route("/")
def home():
    """Renders the main page of the application (e.g., index.html)."""
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def get_recommendations():
    """
    Handles POST requests for movie/web series recommendations.
    Expects JSON input: {"title": "Movie Title", "num": 10, "content_type": "movie/web_series/both"}
    Returns JSON output with recommendations and details of the matched input item.
    """
    data = request.get_json()
    title = data.get("title", "").strip()
    num = int(data.get("num", 10)) # Default to 10 recommendations
    content_type_filter = data.get("content_type", "both").lower() # Default to 'both'

    if not title:
        return jsonify({"status": "error", "message": "Please enter a movie or web series title."}), 400

    # Map frontend 'both' to None for recommender function
    recommender_content_type = None
    if content_type_filter in ['movie', 'webseries']:
        recommender_content_type = content_type_filter

    try:
        # Call the recommend function, which now returns the recommendations DataFrame
        # and the actual title/type of the item matched in your dataset.
        recs, selected_title, selected_type = recommend(
            title,                      # User's raw input title
            df,                         # DataFrame with all content data
            embeddings,                 # Pre-computed embeddings
            num_results=num,
            content_type=recommender_content_type # Filter by type if specified
        )

        if recs is None or recs.empty:
            # If no recommendations are found (e.g., input title not matched, or filters too strict)
            message = f"No recommendations found for '{selected_title}' ({selected_type}) with the given criteria. Try another title or broaden your filter."
            if selected_title == title and selected_type == "N/A": # If the title wasn't even matched
                 message = f"We couldn't find a close match for '{title}'. Please check the spelling or try another title."

            return jsonify({
                "status": "error",
                "message": message,
                "selected_title": selected_title, # Still return what was matched (or input if not found)
                "content_type": selected_type
            })

        # Format recommendations DataFrame into a list of dictionaries for JSON response
        recommendations_list = []
        for _, row in recs.iterrows():
            recommendations_list.append({
                "title": row["title"],
                "genres": row["genres"], # This is a space-separated string, frontend can split/display
                "rating": float(row["vote_average"]) if pd.notnull(row["vote_average"]) else 0.0,
                "popularity": float(row["popularity"]) if pd.notnull(row["popularity"]) else 0.0,
                "platform": row["platform"] if pd.notnull(row["platform"]) else "N/A",
                "seasons": int(row["seasons"]) if pd.notnull(row["seasons"]) else 0,
                "content_type": row["content_type"]
            })

        print(f"\n🎬 Sent {len(recommendations_list)} recommendations to frontend for '{selected_title}'.")
        return jsonify({
            "status": "success",
            "selected_title": selected_title, # The actual title matched in your dataset
            "content_type": selected_type,   # The content type of the matched item
            "recommendations": recommendations_list
        })

    except Exception as e:
        # Catch any unexpected errors during recommendation process
        print(f"❌ An unhandled error occurred during recommendation for '{title}': {str(e)}", exc_info=True) # exc_info for traceback
        return jsonify({"status": "error", "message": "An internal server error occurred. Please try again later."}), 500

# Run the Flask application
if __name__ == "__main__":
    # For local development: debug=True allows hot-reloading and better error messages.
    # For Render deployment: gunicorn will handle running the app, but these defaults are good for local.
    # It's important to bind to '0.0.0.0' for Docker/containerized environments like Render.
    port = int(os.environ.get('PORT', 5050)) # Use PORT env var if set by Render, else default to 5000
    app.run(debug=True, host='0.0.0.0', port=port)