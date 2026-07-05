"""
Build an LLM API wrapper with command line interface.
This wrapper is a personal research assistant. It takes 
questions, calls llm api; and returns structured answers(summary and key points).
We can later improve it enable asking follow up questions
"""

import argparse
from dotenv import load_dotenv
import os
import sys
from huggingface_hub import InferenceClient, InferenceTimeoutError
from huggingface_hub.utils import HfHubHTTPError
import logging
import json

load_dotenv()
logger = logging.getLogger() 

logging.basicConfig(
    filename='app.log',
    format='{asctime} {levelname}: {message}',
    level=logging.DEBUG,
    filemode='w'
)

def valid_json(myjson):
    try:
        return json.loads(myjson)
    except json.JSONDecodeError:
        print("Error:Response is not a valid JSON")
        return None

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
            'content': (
                'You are an expert research assistant. Answer the questions and return structure JSON output with key points and summary')
        }, 
        { 'role': 'user', 'content': user_input }
    ]

    response_format = {
        'type': 'json',
        'value': {
            'properties': {
                'summary': { 'type': 'string', 'description': 'The summary response from the llm api' },
                'key_points': { 'type': 'string', 'description': 'Key points in numbered format' }
            },
            'required': [ 'summary', 'key_points' ]
        }
    }

    try:
        completion = client.chat.completions.create(
            model='meta-llama/Meta-Llama-3-8B-Instruct',
            messages=messages,
            max_tokens=150,
            response_format=response_format
        )
        response = completion.choices[0].message
        parsed_res = valid_json(response.content) # Should return a structured dict {summary: ..., key_Points: ....}
    
        if parsed_res:
            summary = parsed_res.get('summary')
            key_points = parsed_res.get('key_points') # Retrieves key if it exists and None if it doesnt

            if summary is None or key_points is None:
                return None
            return f"The answer to your question is: \nSummary: {summary} \nKey Points: {key_points}"
        
        return None
    except InferenceTimeoutError as e:
        print(f"The model is either unavailable or the request timed out: {e}")

    except HfHubHTTPError as e:
        # Check if response exists first to avoid failure later
        status_code = e.response.status_code if e.response else None

        # Print error according to status codes
        if status_code == 401:
            print("Unauthorized Access: Check API Key permissions")
        elif status_code == 403:
            print("Forbidden: You're not allowed to access this resource")
        elif status_code == 404:
            print("Requested model/resouce not found.")

        elif status_code == 429:
            # Retry logic when rate limit reached
            retry_after = e.response.headers.get('Retry-After')
            if retry_after:
                wait_time = int(retry_after) / 60
                print(f"Rate limited. Try again in {wait_time} minutes")
            else:
                print("Rate limited. Try again later")

        elif status_code is None:
            print(f"Network/connection error occured: {e}")
        else:
            print(f"HTTP error {status_code}: {e}")

    except Exception as e:
        print(f"An unexpected error occured: {e}")

