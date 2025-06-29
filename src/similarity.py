from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity_matrix(tfidf_matrix, bert_embeddings):
    tfidf_sim = cosine_similarity(tfidf_matrix)
    bert_sim  = cosine_similarity(bert_embeddings.cpu().numpy())
    return tfidf_sim, bert_sim
