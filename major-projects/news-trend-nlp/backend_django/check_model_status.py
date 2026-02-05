import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
headers = {"Authorization": f"Bearer {token}"}

models_to_test = [
    "google/flan-t5-base",
    "facebook/bart-large-cnn",
    "gpt2",
    "distilgpt2"
]

print(f"Testing models with token: {token[:5]}...")

for model in models_to_test:
    url = f"https://api-inference.huggingface.co/models/{model}"
    try:
        # Just check status with a lightweight request
        response = requests.get(url, headers=headers, timeout=5)
        print(f"{model}: {response.status_code}")
    except Exception as e:
        print(f"{model}: Error {e}")
