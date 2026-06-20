import os
import time
from google import genai
from dotenv import load_dotenv


load_dotenv()


client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


MODEL_NAME = "gemini-2.0-flash-lite"



def ask_gemini(prompt):

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )


        if response is None:
            return "Gemini returned empty response."


        if not hasattr(response, "text"):
            return "Gemini response missing text."


        if response.text is None or response.text.strip()=="":
            return "Gemini generated empty report."


        return response.text


    except Exception as e:

        error = str(e)


        if "503" in error or "UNAVAILABLE" in error:

            time.sleep(5)

            try:

                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt
                )

                if response.text:
                    return response.text


            except Exception:
                pass


        return """
## ⚠️ AI Report Temporarily Unavailable

Solar prediction completed successfully.

Please check:
- AC power output
- Irradiation
- Temperature
- SHAP feature impact

If abnormal behaviour is detected,
contact solar maintenance technicians.
"""
