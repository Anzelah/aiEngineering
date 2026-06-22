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
    
    try:
        load_dotenv()
        env_var = os.environ
        print(f"The environmental variables are: {env_var}")
    except FileNotFoundError as file_err:
        print(f"The file requested doesn't exist: {file_err}")
        return None
    except KeyError as key_err:
        print(f"Environmental variable doesn't exist: {key_err}")
        return None
    except OSError as e:
        print(f"A system-level error occurred: {e}")
        return None
    