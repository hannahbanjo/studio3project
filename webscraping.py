from bs4 import BeautifulSoup
import requests
import pandas as pd
import pdfplumber
from io import BytesIO
import re

csv = pd.read_csv('websites.csv')

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
                print(f"Preview from page: {txt[:50]}")
    return txt

for index, row in csv.iterrows():
    if row["type"] == "website":
       cleaned_text = scrape_website(row["link"])
       csv.loc[index, 'cleaned_text'] = cleaned_text
       
    if row["type"] == "pdf":
        cleaned_text = scrape_pdf(row["link"])
        csv.loc[index, 'cleaned_text'] = cleaned_text