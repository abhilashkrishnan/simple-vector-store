import numpy as np

def cosine_similarity(store_embeddings, query_embedding, top_k):
    # Compute dot product
    dot_product = np.dot(store_embeddings, query_embedding)
    
    # Compute magnitudes
    magnitude_a = np.linalg.norm(store_embeddings, axis=1)
    magnitude_b = np.linalg.norm(query_embedding)
    
    # Compute cosine similarity
    similarity = dot_product / (magnitude_a * magnitude_b)
    
    # Sort indices by similarity in descending order
    sim = np.argsort(similarity)[::-1]  # Reverse for descending order
    
    # Retrieve the top_k indices
    top_k_indices = sim[:top_k]  # Select the first `top_k` indices
    
    return top_k_indices
