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
    """Validate output is json object"""
    try:
        return json.loads(myjson)
    except json.JSONDecodeError:
        print("Error:Response is not a valid JSON")
        return None

def validate_schema(data):
    """Ensure required keys(summary and keypoints) exist"""
    if not isinstance(data, dict):
        return None
    
    summary = data.get('summary')
    key_points = data.get('key_points') # Retrieves key if it exists and None if it doesnt
    
    if summary is None or key_points is None:
        return None
    
    return data

def format_output(data):
    """Format final response to send to user"""

    return f"""
    The answer to your question is: \n
    Summary: {data['summary']} \n
    Key Points: {data['key_points']}"""


def parse_input():
    """Parse user input from CLI using Argparse"""
    parser = argparse.ArgumentParser(
        description='Send user input to LLM', 
        suggest_on_error=True
    )

    parser.add_argument('user_input', type=str, help='Input string')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    # Check for empty strings
    if not args.user_input.strip():
        parser.error(f"Input cannot be empty")
    
    if args.verbose:
        print(f"User input received succesfully")

    return args.user_input


def call_api(user_input):
    """Call an llm wrapper with our user_input as the input"""
    api_key = os.getenv('HF_API_KEY')
    if not api_key:
        sys.exit("Error: Missing API key")

    # Call LLM api. To use chat completions
    client = InferenceClient(api_key=api_key)

    messages = [
        {
            'role': 'system',
            'content': (
                'You are an expert research assistant. return structured JSON with summary and key_points')
        }, 
        { 'role': 'user', 'content': user_input }
    ]

    response_format = {
        'type': 'json',
        'value': {
            'properties': {
                'summary': { 'type': 'string', 'description': 'Summary response from the llm api' },
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
        response_content = completion.choices[0].message.content

        # Validate it's json and parse it
        parsed = valid_json(response_content) # Should return a structured dict {summary: ..., key_Points: ....}
        if not parsed:
            return None
    
        # Validate it has content
        validated = validate_schema(parsed)

        #Format output
        final_res = format_output(validated)
        return final_res
    
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

        else:
            print(f"HTTP error {status_code}: {e}")
    except Exception as e:
        print(f"An unexpected error occured: {e}")


def main():
    """Main function"""
    # Get user input
    user_input = parse_input()
    result = call_api(user_input)

    if result:
        print(result)
    else:
        print("Failed to get a valid response")
    return result

if __name__ == "__main__":
    main()