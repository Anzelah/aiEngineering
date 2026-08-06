"""
Build an LLM API wrapper with command line interface.
This wrapper is a personal research assistant. It takes 
questions, calls llm api; and returns structured answers(summary and key points).
We can later improve it enable asking follow up questions
"""

import os
import sys
import json
import logging
import argparse
from dotenv import load_dotenv
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import InferenceClient, InferenceTimeoutError


load_dotenv()

class APIRequestError(Exception):
    """Raised when API call fails"""
    pass

logging.basicConfig(
    filename='app.log',
    format='{asctime} {levelname}: {message}',
    level=logging.DEBUG,
    filemode='a',
    style='{'
)

logger = logging.getLogger(__name__) 


def valid_json(myjson):
    """Validate output is json object. 
    Returns a structured dict {summary: ..., key_Points: ....}"""

    try:
        return json.loads(myjson)
    except json.JSONDecodeError:
        logger.error("Invalid JSON received")
        raise TypeError("Response is not a valid JSON")


def validate_schema(data):
    """Ensure required keys(summary and keypoints) exist, that response has content"""
    if not isinstance(data, dict):
        logger.warning("Wrong/unexpected response object")
        raise TypeError("Response is not a dictionary")
    
    summary = data.get('summarized_answer')
    key_points = data.get('key_points') 
    
    if summary is None or key_points is None:
        logger.warning("Missing required fields in response")
        raise KeyError
    
    return data


def format_output(data):
    """Format final response to send to user"""

    return f"""
    Summary: {data['summarized_answer']} \n
    Key Points: {data['key_points']}"""


def parse_input():
    """Parse user input from CLI using Argparse"""
    logger.info("Parsing user input")

    parser = argparse.ArgumentParser(
        description='Send user input to LLM'
    )

    parser.add_argument('user_input', type=str, help='Input string')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')

    args = parser.parse_args()

    # Check for empty strings
    if not args.user_input.strip():
        logger.error("Empty input provided")
        parser.error("Input cannot be empty")
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("User input received")
    return args.user_input


def call_api(user_input):
    """Call an llm wrapper with our user_input as the input"""
    api_key = os.getenv('HF_API_KEY')
    if not api_key:
        logger.error("Missing api key")
        raise APIRequestError("Missing required API key")

    # Call LLM api. To use chat completions
    client = InferenceClient(api_key=api_key)

    messages = [
        {
            'role': 'system',
            'content': (
                """You are an intent classifier. Analyze the user texts, then classify them into appropriate lables. 
                The labels are: billing_issue, refund_request, technical_issue, general_question and other.
                Examples of text fitting as other: "Hey there", "Just testing", "Hello???", "This is random".
                If unsure, classify as other rather than guessing
                Return only valid structured JSON with text_intent without preamble""")
        }, 
        { 'role': 'user', 'content': user_input }
    ]

    response_format = {
        'type': 'json_schema',
        'value': {
            'properties': {
                'text_intent': { 'type': 'string', 'description': 'Text intent from the llm api' }
                #'summary': { 'type': 'string', 'description': 'Summary response from the llm api' },
                #'key_points': { 'type': 'string', 'description': 'Key points in numbered format' }
            },
            'required': [ 'text_intent' ]
        }
    }
    logger.info("Preparing API request")

    try:
        completion = client.chat.completions.create(
            model='Qwen/Qwen2.5-7B-Instruct',
            messages=messages,
            max_tokens=500,
            response_format=response_format
        )
        response_content = completion.choices[0].message.content
        print(f"LLM Respose is: {response_content}" ) # For Debugging
        logger.debug(f"Raw LLM Api response: {response_content}")

        parsed = valid_json(response_content)
        validated = validate_schema(parsed)

        #Format output
        return format_output(validated)
    
    except InferenceTimeoutError as e:
        logger.error(f"The model is either unavailable or the request timed out: {e}")
        raise APIRequestError("API request timeout")

    except HfHubHTTPError as e:
        # Check if response exists first to avoid failure later
        status_code = e.response.status_code if e.response else None

        if status_code == 401:
            logger.error("Unauthorized Access: Check API Key permissions")
            raise APIRequestError("Unauthorized access")

        elif status_code == 403:
            logger.error("Forbidden access. You're not allowed access to this resource")
            raise APIRequestError("Forbidden access")

        elif status_code == 404:
            logger.error("Resource not found.")
            raise APIRequestError("Requested model/resouce not found.")

        elif status_code == 429:
            # Retry logic when rate limit reached
            retry_after = e.response.headers.get('Retry-After')
            if retry_after:
                wait_time = int(retry_after) / 60
                logger.error(f"Rate limited. Try again in {wait_time} minutes")
                raise APIRequestError(f"Rate limited. Try again in {wait_time} minutes")

            else:
                logger.error("Rate limited. Try again later")
                raise APIRequestError("Rate limited. Try again later")

        else:
            logger.error(f"HTTP error {status_code}: {e}")
            raise APIRequestError(f"http error {status_code}: {e}")


    except Exception as e:
        logger.exception(f"An unhandled error occured when calling API: {e}")
        raise


def main():
    """Main function"""
    try: 
        logger.info("Application started")

        # Get user input
        user_input = parse_input()
        result = call_api(user_input)

        print(result)
        logger.info("Application completed successfully")
        return result

    except TypeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    except KeyError:
        print("Missing required fields in response")
        sys.exit(1)
    
    except APIRequestError as e:
        print(f"API error: {e}")
        sys.exit(1)

    except Exception as e:
        logger.exception("An unexpected error occured in main")
        sys.exit(1)

if __name__ == "__main__":
    main()