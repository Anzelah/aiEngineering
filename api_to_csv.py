"""
A function that calls an API and pulls data from it,
then write this data in to a SCV file
"""

import requests
import csv

def pull_data():
    """call and api and pull its data"""


    try:
        url = "https://jsonplaceholder.typicode.com/users"
        res = requests.get(url, timeout=5)
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
    """Write API data into a csv"""
    data = pull_data()
    if not data:
        print("No data returned from API")
        return None
    
    if not isinstance(data, list):
        print(f"Unexpected data format: ", type(data))
        return
    
    with open("output.csv", "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csv_file, delimiter=' ', fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
if __name__ == "__main__":
    write_to_csv()