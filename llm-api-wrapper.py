"""
Build an LLM API wrapper with command line interface.
This wrapper is a support simple text analyzer. We'll do the support ticket analyzer later'
"""

import argparse
from dotenv import load_dotenv
import os
import sys
from huggingface_hub import InferenceClient


def parse_input():
    """Parse user input from CLI using Argparse"""
    parser = argparse.ArgumentParser(description='Take user input text and passes in onto another function', suggest_on_error=True)
    parser.add_argument('user_input', type=str, help='The string to send to api')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # Check for empty strings
    if not args.user_input.strip():
        parser.error(f"Input cannot be empty")
    
    if args.verbose:
        print(f"User input received succesfully")

    return args.user_input


def call_api():
    """Call an llm wrapper with our user_input as the input"""
    load_dotenv()
    api_key = os.getenv('HF_API_KEY')
    if not api_key:
        sys.exit("Error: Missing API key")

    # Get user input
    user_input = parse_input()

    # Call LLM api. To use chat completions
    client = InferenceClient(api_key=api_key)
    messages = [
        {
            'role': 'system',
            'content': ('You are an expert assistant')
        }, 
        { 'role': 'user', 'content': user_input }
    ]

    response_format = {
        'type': 'json',
        'value': {
            'properties': {
                'summary': { 'type': 'string', 'description': 'The summary response from the llm api' },
                'sentiment': { 'type': 'string', 'description': 'What is the tone of the user input?' }
            },
            'required': [ 'summary', 'sentiment' ]
        }
    }

    completion = client.chat.completions.create(
        model='meta-llama/Meta-Llama-3-8B-Instruct',
        messages=messages,
        max_tokens=150,
        response_format=response_format
    )





def json_output():
    """Enforce JSON response format via the prompt"""
    pass


def errors_and_rate_limit():
    pass



def log_results():
    pass

