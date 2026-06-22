"""
This function script loads secrets from an
.env file and handles error cleanly
"""

from dotenv import load_dotenv
import os

def load_secrets():
    """A file that loads .env file"""
    if not os.path.exists(".env"):
        print("The .env file doesn't exist")
        return
    
    load_dotenv()
    api_key = os.getenv("API_KEY")

    