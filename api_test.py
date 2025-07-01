import os, requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ELEVENLABS_API_KEY")
resp = requests.get("https://api.elevenlabs.io/v1/voices",
                    headers={"xi-api-key": API_KEY})
print(resp.status_code, resp.text)
