from flask import Flask, render_template, request, jsonify
from src.data_loader import load_and_clean_data
from src.features import combine_features
from src.recommender import generate_embeddings, recommend
import os
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

def load_everything():
    print("🚀 Testing minimal load: just DataFrame, no embeddings")

    df = pd.read_csv("data/movies_metadata.csv")
    df = df.head(50)  # ✅ Only load 50 rows temporarily

    df["combined_features"] = df["title"].fillna("")  # Dummy placeholder
    embeddings = np.random.rand(50, 384).astype(np.float32)  # Dummy embeddings

    return df, embeddings

# --- ROUTES ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def get_recommendations():
    df, embeddings = load_everything()

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
            msg = f"No recommendations found for '{selected_title}' ({selected_type})."
            if selected_title == title and selected_type == "N/A":
                msg = f"No close match found for '{title}'."

            return jsonify({
                "status": "error",
                "message": msg,
                "selected_title": selected_title,
                "content_type": selected_type
            })

        recommendations = []
        for _, row in recs.iterrows():
            recommendations.append({
                "title": row.get("title", ""),
                "genres": row.get("genres", ""),
                "rating": float(row.get("vote_average", 0.0)),
                "popularity": float(row.get("popularity", 0.0)),
                "platform": row.get("platform", "N/A"),
                "seasons": int(row.get("seasons", 0)),
                "content_type": row.get("content_type", "")
            })

        print(f"🎬 Sent {len(recommendations)} recommendations for '{selected_title}'")
        return jsonify({
            "status": "success",
            "selected_title": selected_title,
            "content_type": selected_type,
            "recommendations": recommendations
        })

    except Exception as e:
        print(f"❌ Recommendation Error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Internal server error occurred."}), 500

# --- RUN LOCALLY ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=True, host="0.0.0.0", port=port)
