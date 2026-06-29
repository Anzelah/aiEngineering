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
        print(f"Success: The file was loaded succesfully!")
    except FileNotFoundError:
        print(f"Error: The file doesnt exist")
    except pd.errors.EmptyDataError:
        print(f"The file is completely empty")
    except Exception as e:
        print(f"An unexpected error occured: {e}")
    print(df)

    # Load a pretained Sentence Transformer model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Generate embeddings from our input data
    embeddings = model.encode_query(df, prompt_name="query", show_progress_bar=True)
    print(f"A list of numbers representing the sentence meaning: {embeddings}")

    # Compute similarities using cosin similarity
    similarities = util.cos_sim(embeddings, embeddings)
    print(similarities)


if __name__ == "__main__":
    generate_embeddings()
    