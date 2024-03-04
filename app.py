from sentence_transformers import SentenceTransformer
import numpy as np

def text_embedding(text) -> None:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(text, normalize_embeddings=True)

phrase= 'apple is a fruit'
embedding1= text_embedding(phrase)

phrase2= 'apple iPhone is expensive'
embedding2= text_embedding(phrase2)

len(embedding1), len(embedding2)

def vector_similarity(vec1, vec2):
    return np.dot(np.squeeze(np.array(vec1)), np.squeeze(np.array(vec2)))

print(vector_similarity(embedding1, embedding2))