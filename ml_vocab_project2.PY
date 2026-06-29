"""
Generate embeddings and compute similarity
"""

from sentence_transformers import SentenceTransformer
import pandas as pd

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
    