"""
Generate embeddings and compute similarity
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os

def generate_embeddings():
    """Use pretained model SentenceTransformers to generate vectors"""
    # load data 
    try:
        df = pd.read_csv('Mental_Health_FAQ.csv')
        corpus = df['Questions'].tolist()
        answers = df['Answers'].tolist()
        print(f"Success: The file was loaded succesfully!")
    except FileNotFoundError:
        print(f"Error: The file doesnt exist")
        return None
    except pd.errors.EmptyDataError:
        print(f"The file is completely empty")
        return None
    except Exception as e:
        print(f"An unexpected error occured: {e}")
        return None
    
    # Load a pretained Sentence Transformer model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Generate embeddings from our input data if they dont exist
    file_path = "embeddings.npy"
    if os.path.exists(file_path):
        corpus_embeddings = np.load(file_path)
    else:
        corpus_embeddings = model.encode(corpus, show_progress_bar=True)
        np.save(file_path, corpus_embeddings)

    # Add a query(real_world scenario) and create embeddings for it 
    query = ['How do i reduce anxiety?']
    query_embedding = model.encode(query)

    # Compute similarity scores finding sentences in corpus similar to our query
    similarity_scores = util.cos_sim(corpus_embeddings, query_embedding)

    # Extract the top 5 similar sentences in our similarity_score row
    scores, indices = similarity_scores.topk(k=5, dim=0)
    for score, index in zip(scores, indices):
        print(f"The top five similar queries are: {answers[index.item()]} -> score: {score:.4f}") 


if __name__ == "__main__":
    generate_embeddings()
    