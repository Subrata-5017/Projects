# app.py (Gradio version)
import gradio as gr
from src.data_loader import load_and_clean_data
from src.features import combine_features
from src.recommender import generate_embeddings, recommend

import numpy as np
import os

# Load and prepare data and embeddings
df = load_and_clean_data()
df['combined_features'] = combine_features(df)

embedding_path = "embeddings.npy"
if os.path.exists(embedding_path):
    embeddings = np.load(embedding_path)
else:
    embeddings = generate_embeddings(df["combined_features"])
    np.save(embedding_path, embeddings)

def recommend_ui(title, num, content_type):
    recs, matched_title, matched_type = recommend(title, df, embeddings, int(num), content_type)
    if recs is None or recs.empty:
        return f"No recommendations for '{title}' ({content_type})"
    return "\n".join(recs['title'].tolist())

interface = gr.Interface(
    fn=recommend_ui,
    inputs=[
        gr.Textbox(label="Title"),
        gr.Slider(minimum=1, maximum=20, step=1, label="Number of Recommendations", value=10),
        gr.Dropdown(choices=["movie", "webseries", "both"], label="Content Type", value="both")
    ],
    outputs="text",
    title="🎬 Movie & Web Series Recommender"
)

interface.launch()
