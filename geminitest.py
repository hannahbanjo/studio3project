import os
from google import genai

# API_KEY = os.environ["API_KEY"]
API_KEY = "AIzaSyDLxtpeLSf_zO3C0zMIwvc6Ml8UPfS8Mo8"
client = genai.Client(api_key=API_KEY)

response = client.models.generate_content(
    model = "gemini-2.5-flash",
    contents = "Generate a sample 50 word paragraph about the benefits of web scraping."
)

print(response.text)
