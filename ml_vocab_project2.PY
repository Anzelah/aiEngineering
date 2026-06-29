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
        return None
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
    embeddings = model.encode(data, show_progress_bar=True)

    # Compute similarities using cosin similarity
    similarities = util.cos_sim(embeddings, embeddings)
    print(similarities)   


if __name__ == "__main__":
    generate_embeddings()
    