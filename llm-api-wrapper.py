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
        parsed = json.loads(myjson)
        return parsed
    except json.JSONDecodeError as e:
        print(f"The object provided is not a valid JSON")
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
                'key points': { 'type': 'string', 'description': 'Key points in numbered format' }
            },
            'required': [ 'summary', 'key points' ]
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
        parsed_res = valid_json(response.content)
        print(f"This is the parsed Response(to send to user): {parsed_res}")
        
        if parsed_res:
            return f"The summarized answer to your questions is: {parsed_res.summary}. \nKey points include: {parsed_res.key_points}"

        return None
    except InferenceTimeoutError as e:
        print(f"The model is either unavailable or the request timed out: {e}")

    except HfHubHTTPError as e:
        # Check if response exists first to avoid failure later
        if e.response:
            status_code = e.response.status_code
        else:
            status_code = None

        # Print error according to status codes
        if status_code == 401:
            print("Unauthorized access. Check your api token permissions")
        elif status_code == 403:
            print("Forbidden. You are not allowed to access this resource")
        elif status_code == 404:
            print("Requested resource doesn't exist. It might've been moved elsewhere or a mispelled link")

        elif status_code == 429:
            # Retry logic when rate limit reached
            retry_after = e.response.headers.get('Retry-After')
            if retry_after:
                wait_time = int(retry_after) / 60
                print(f"Too many requests. Try again after {wait_time} minutes")
            else:
                print("Too many requests. Try again after a while")

        elif status_code is None:
            print(f"HTTP error occurred but no response was returned: {e}")
        else:
            print(f"HTTP error {status_code} occured: {e}")

    except Exception as e:
        print(f"An unexpected error occured: {e}")





def json_output():
    """Enforce JSON response format via the prompt"""
    pass


def errors_and_rate_limit():
    pass



def log_results():
    pass

