"""
This function script loads secrets from an
.env file and handles error cleanly
"""

from dotenv import dotenv_values
import os


def load_secrets():
    """A file that loads .env file"""
    if not os.path.exists(".env"):
        print("The .env file doesn't exist")
        return None
    
    try:
        config = dotenv_values(".env")
        if not config:
            print(f"Failed to load env variables. The .env file may be empty or invalid")
            return None
        
        for key, value in config.items():
            print(f"{key}: {value}")
        return config

    except OSError as os_err:
        print(f"A system-level error occurred: {os_err}")
        return None
    except Exception as e:
        print(f"An unexpected error occured: {e}")
        return None
    

if __name__ == "__main__":
    load_secrets()