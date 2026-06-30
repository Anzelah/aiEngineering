"""
Tokenize a sentence with hugging face and inspect the tokens
"""

from transformers import AutoTokenizer

def tokenize_sentences():
    """Use pretrained tokenizer in hugging face"""

    # Load tokenizer of chosen model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Input your sentence
    sentence = 'i am happy'

    # 3. See the actual tokens and IDs
    encoded = tokenizer(sentence)
    print("TOKEN TO ID MAPPING:")
    print("-" * 25)
    for idx in encoded["input_ids"]:
        # Decode each individual ID back to its token string
        token_text = tokenizer.decode([idx])
        print(f"Token: '{token_text}'  ->  ID: {idx}")

if __name__ == "__main__":
    tokenize_sentences()