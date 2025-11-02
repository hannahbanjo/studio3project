import os
import google.generativeai as genai
from dotenv import load_dotenv

json_data = "https://github.com/hannahbanjo/studio3project/blob/main/websites.json"

load_dotenv()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content([
    json_data,
    "For each value in the JSON object, can you identify if the article is fraud related based on the 'leaned_text' value. Just respond with a simple 'Fraud-Related' or 'Not Fraud-Related' for each website."
])
print(response.text)