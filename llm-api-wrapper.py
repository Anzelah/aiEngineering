"""
Build an LLM API wrapper with command line interface.
This wrapper is a support simple text analyzer. We'll do the support ticket analyzer later'
"""

import argparse


def user_input():
    """Parse user input from CLI using Argparse"""
    parser = argparse.ArgumentParser(description='Take user input text and passes in onto another function', suggest_on_error=True)
    parser.add_argument('user_question', type=str, help='The string to send to api')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # Check for empty strings
    if not args.user_question.strip():
        parser.error(f"Input cannot be empty")
    
    if args.verbose:
        print(f"User input received succesfully")

    return args.user_question


def call_api():
    pass


def json_output():
    """Enforce JSON response format via the prompt"""
    pass


def errors_and_rate_limit():
    pass



def log_results():
    pass

