"""
A function that calls an API and pulls data from it,
then write this data in to a SCV file
"""

import requests
import csv

def pull_data():
    """call and api and pull its data"""


    try:
        url = "http://worldtimeapi.org/api/timezone/"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occured: {http_err}")
        return
    except requests.exceptions.Timeout as time_err:
        print(f"The request timed out: {time_err}")
        return
    except requests.exceptions.TooManyRedirects as redir_err:
        print(f"Too many redirects: {redir_err}")
        return
    else:
        # parse the JSON response into a dictionary
        data = res.json()
        return data

def write_to_csv():
    """"Write API data into a csv"""
    data = pull_data()
    if data is None:
        return None
    
    if isinstance(data, dict):
        with open("output.csv", "w") as csv_file:
            fieldnames = [ "Name", "Age", "Occupation"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    elif (isinstance(data, list)):
        with open("output.csv", "w", newline='') as csv_file:
            fieldnames = [ "Name", "Age", "Occupation"]
            writer = csv.writer(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    else:
        return 