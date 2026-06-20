import os
import time
from google import genai
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Gemini client
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


# Stable free-tier model
MODEL_NAME = "gemini-2.0-flash-lite"



def ask_gemini(prompt):

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        return response.text


    except Exception as e:

        error = str(e)

        # Handle temporary overload
        if "503" in error or "UNAVAILABLE" in error:

            time.sleep(3)

            try:

                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt
                )

                return response.text


            except Exception:

                return """
## ⚠️ AI Report Temporarily Unavailable

Gemini service is currently busy.

Solar prediction results are still available:

Please check:
- AC power output
- irradiation
- temperature
- SHAP feature impact

Try again after a few seconds.
"""


        return f"""
## ⚠️ AI Analysis Error

{error}
"""
