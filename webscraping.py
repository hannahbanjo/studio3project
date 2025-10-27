from bs4 import BeautifulSoup
import requests
import pandas as pd
import pdfplumber
from io import BytesIO
import re
import csv

website_csv = pd.read_csv('websites.csv')

def scrape_website(link):
    print(f"Scraping website: {link}")
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def scrape_pdf(link):
    print(f"Scraping PDF: {link}")
    resp = requests.get(link)
    file = BytesIO(resp.content)
    all_text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                all_text.append(txt)
    return all_text

for index, row in website_csv.iterrows():
    if row["type"] == "website":
        cleaned_text = scrape_website(row["link"])
        website_csv.loc[index, 'cleaned_text'] = cleaned_text
        
    if row["type"] == "pdf":
        cleaned_text = scrape_pdf(row["link"])
        website_csv.loc[index, 'cleaned_text'] = cleaned_text

website_csv.to_csv("websites.csv", index=False)
