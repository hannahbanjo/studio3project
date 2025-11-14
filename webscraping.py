from bs4 import BeautifulSoup
import requests
import pandas as pd
import pdfplumber
from io import BytesIO
import re
import os
from dotenv import load_dotenv

load_dotenv()

def scrape_website(link):
    print(f"Scraping website: {link}")
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.extract()
    text = soup.get_text()
    date = soup.select_one('span.date')
    date = date.get_text(strip=True) if date else None
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text, date

date_pattern = [
    r"[A-Z][a-z]+ \d{1,2}, \d{4}",     # December 4, 2024
    r"[A-Z][a-z]{2} \d{1,2}, \d{4}",  # Dec 4, 2024
    r"\d{4}-\d{2}-\d{2}",             # 2024-12-04
    r"\d{1,2}/\d{1,2}/\d{4}",         # 12/04/2024
]

def extract_date(text):
    for pattern in date_pattern:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None

def scrape_pdf(link):
    print(f"Scraping PDF: {link}")
    resp = requests.get(link)
    file = BytesIO(resp.content)
    all_text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                clean_page = txt
                all_text.append(clean_page)
    full_text = " ".join(all_text)
    date = extract_date(full_text)

    return full_text, date

website_csv = pd.read_csv("websites.csv")

cleaned_texts = []
dates = []

for index, row in website_csv.iterrows():
    if row["type"] == "website":
        cleaned_text, date = scrape_website(row["link"])
        
    if row["type"] == "pdf":
        cleaned_text, date = scrape_pdf(row["link"])

    cleaned_texts.append(cleaned_text)
    dates.append(date)

website_csv["cleaned_text"] = cleaned_texts
website_csv["date"] = dates
website_csv.to_csv("websites.csv", index=False)

df = pd.read_csv("websites.csv")
json_data = df.to_json("websites.json", orient='records', indent=2)