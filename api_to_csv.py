"""
A function that calls an API and pulls data from it,
then write this data in to a SCV file
"""

import requests
import csv

def pull_data():
    """call and api and pull its data"""


    try:
        res = requests.get("url", timeout=5)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occured: {http_err}")
        return None
    except requests.exceptions.Timeout as time_err:
        print(f"The request timed out: {time_err}")
        return None
    except requests.exceptions.TooManyRedirects as redir_err:
        print(f"Too many redirects: {redir_err}")
        return None
    

def write_to_csv():
    """"Write API data into a csv"""
    data = pull_data()
    if not data:
        return None
    
    with open("output.csv", "w", newline='') as csv_file:
        fieldnames = ["Name", "Age", "Occupation"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    