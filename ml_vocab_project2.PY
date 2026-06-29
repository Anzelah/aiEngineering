"""
Generate embeddings and compute similarity
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, util

def generate_embeddings():
    """Use pretained model SentenceTransformers to generate vectors"""
    # load data 
    try:
        df = pd.read_csv('Mental_Health_FAQ.csv')
        data = df['Questions'].tolist()
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
    
    # Generate embeddings from our input data
    data_embeddings = model.encode(data, show_progress_bar=True)

    query = ['How do i reduce anxiety?']
    query_embeddings = model.encode(query)
    # Compute similarities using cosin similarity
    similarity_scores = util.cos_sim(data_embeddings, query_embeddings)
    for i in similarity_scores:
        print(i)   


if __name__ == "__main__":
    generate_embeddings()
    