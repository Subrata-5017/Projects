# app.py (Optimized for Deployment)
from flask import Flask, render_template, request, jsonify
from src.data_loader import load_and_clean_data
from src.features import combine_features
from src.recommender import generate_embeddings, recommend, clean_title

import os
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

# Lazy-loaded globals
df = None
embeddings = None

def initialize_if_needed():
    global df, embeddings

    if df is not None and embeddings is not None:
        return  # Already initialized

    print("🚀 Lazy Initialization: Loading data and embeddings...")

    # Load and preprocess data
    try:
        df_loaded = load_and_clean_data()
        df_loaded['combined_features'] = combine_features(df_loaded)
        print(f"✅ Loaded {len(df_loaded)} items with combined features.")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        exit()

    # Handle embeddings
    embedding_path = "Movie_Recommender/embeddings.npy"
    emb = None
    if os.path.exists(embedding_path):
        try:
            emb = np.load(embedding_path)
            if emb.shape[0] != len(df_loaded):
                print("⚠️ Embeddings count mismatch — regenerating...")
                emb = generate_embeddings(df_loaded["combined_features"])
                np.save(embedding_path, emb)
        except Exception as e:
            print(f"❌ Failed to load cached embeddings: {e}")
            emb = generate_embeddings(df_loaded["combined_features"])
            np.save(embedding_path, emb)
    else:
        print("🔧 No cached embeddings — generating...")
        emb = generate_embeddings(df_loaded["combined_features"])
        np.save(embedding_path, emb)

    globals()["df"] = df_loaded
    globals()["embeddings"] = emb
    print(f"✅ Embeddings ready: {emb.shape}")

# --- ROUTES ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def get_recommendations():
    initialize_if_needed()

    data = request.get_json()
    title = data.get("title", "").strip()
    num = int(data.get("num", 10))
    content_type_filter = data.get("content_type", "both").lower()

    if not title:
        return jsonify({"status": "error", "message": "Please enter a title."}), 400

    recommender_content_type = None
    if content_type_filter in ['movie', 'webseries']:
        recommender_content_type = content_type_filter

    try:
        recs, selected_title, selected_type = recommend(
            title,
            df,
            embeddings,
            num_results=num,
            content_type=recommender_content_type
        )

        if recs is None or recs.empty:
            message = f"No recommendations found for '{selected_title}' ({selected_type})."
            if selected_title == title and selected_type == "N/A":
                message = f"No close match for '{title}'. Check spelling."

            return jsonify({
                "status": "error",
                "message": message,
                "selected_title": selected_title,
                "content_type": selected_type
            })

        results = []
        for _, row in recs.iterrows():
            results.append({
                "title": row["title"],
                "genres": row["genres"],
                "rating": float(row.get("vote_average", 0.0)),
                "popularity": float(row.get("popularity", 0.0)),
                "platform": row.get("platform", "N/A"),
                "seasons": int(row.get("seasons", 0)),
                "content_type": row["content_type"]
            })

        print(f"🎬 Sent {len(results)} recommendations for '{selected_title}'.")
        return jsonify({
            "status": "success",
            "selected_title": selected_title,
            "content_type": selected_type,
            "recommendations": results
        })

    except Exception as e:
        print(f"❌ Internal error during recommendation: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Internal server error."}), 500

# --- Run in Local Dev ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=True, host="0.0.0.0", port=port)
