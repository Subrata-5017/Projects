# src/features.py
import pandas as pd
import re

def clean_text(text):
    """
    Cleans text for embedding. This version is specifically for the combined features
    that go into the SentenceTransformer. It removes non-alphanumeric,
    converts to lowercase, and standardizes spaces.
    """
    if isinstance(text, str):
        # Remove characters that are not letters, numbers, or spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Replace multiple spaces with a single space and strip leading/trailing spaces
        return re.sub(r'\s+', ' ', text).strip()
    return ''

def combine_features(data):
    """
    Combines specified textual features into a single string for embedding.
    Ensures all input features are treated as strings and are cleaned.
    """
    # These fields are crucial for semantic understanding by the BERT model
    fields = ['keywords', 'cast', 'genres', 'director', 'overview', 'content_rating', 'platform', 'content_type']

    def combine_row(row):
        # Retrieve each field safely, convert to string, clean, and then join
        combined = " ".join([clean_text(str(row.get(field, ''))) for field in fields])
        return combined

    # Apply the combining function row-wise
    return data.apply(combine_row, axis=1)