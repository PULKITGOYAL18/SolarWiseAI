import os
from google import genai
from dotenv import load_dotenv
# Load .env
load_dotenv()
# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)
# Keep your same model
MODEL_NAME = "gemini-flash-lite-latest"
def ask_gemini(prompt):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text
