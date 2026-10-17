import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("\n[ERROR] GEMINI_API_KEY not found in .env!")
    print("Please open the .env file in the project root and add:")
    print("GEMINI_API_KEY=your_actual_key_here\n")
    exit(1)

# Gemma 4 model ID from Google AI Studio
MODEL_ID = "gemma-4-26b-a4b-it"

url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={api_key}"

payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": "Write a 1-sentence concise security alert announcing that an unattended backpack has been detected in Terminal 2."
                }
            ]
        }
    ]
}

print(f"Connecting to Gemini API with model '{MODEL_ID}'...")

try:
    response = requests.post(url, json=payload, timeout=25)
    if response.status_code == 200:
        data = response.json()
        alert_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        print("\n=======================================================")
        print(" SUCCESS! Gemma 4 generated the alert:")
        print(f"\"{alert_text}\"")
        print("=======================================================\n")
    else:
        print(f"\n[API ERROR {response.status_code}]")
        print(response.text)
        print("\nTip: If this model ID returns 404, check Google AI Studio for available Gemma 4 identifiers.")
except Exception as e:
    print(f"\n[NETWORK ERROR] {e}")

