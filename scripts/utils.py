import requests
import xml.etree.ElementTree as ET
import csv

def fetch_arxiv_data(query, max_results=100):
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query={query}&start=0&max_results={max_results}"
    response = requests.get(base_url + search_query)
    return response.text

data = fetch_arxiv_data("all:deep learning")

# Parsing XML response
root = ET.fromstring(data)
entries = root.findall('{http://www.w3.org/2005/Atom}entry')

with open("arxiv_data.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "title", "authors", "categories", "abstract", "created", "updated"])
    for entry in entries:
        id = entry.find('{http://www.w3.org/2005/Atom}id').text
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
        categories = [category.attrib['term'] for category in entry.findall('{http://arxiv.org/schemas/atom}category')]
        abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text
        created = entry.find('{http://www.w3.org/2005/Atom}published').text
        updated = entry.find('{http://www.w3.org/2005/Atom}updated').text
        writer.writerow([id, title, ','.join(authors), ','.join(categories), abstract, created, updated])