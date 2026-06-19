from agents.gemini_core import ask_gemini
import json
class InputAgent:
    def extract(
        self,
        date_val,
        time_val,
        temp_val,
        weather_type,
        day_history
    ):
        day_of_year = date_val.timetuple().tm_yday
        prompt = f"""
You are a solar power input preparation expert.
The ML model predicts AC POWER output (Watts).
The model input features are:
- AMBIENT_TEMPERATURE
- MODULE_TEMPERATURE
- IRRADIATION
- HOUR
- MINUTE
- DAY_OF_YEAR
User Conditions:
Weather:
{weather_type}
Time:
{time_val}
Previous Context:
{day_history}
Generate realistic solar parameters.
Rules:
1. Night condition:
If hour < 6 OR hour >= 19:
IRRADIATION = 0.0
AC Power generation should be considered unavailable.
2. Rain / Heavy Cloud:
IRRADIATION range:
0.01 - 0.20
3. Cloudy:
IRRADIATION range:
0.20 - 0.50
4. Normal Sunny:
IRRADIATION range:
0.50 - 0.85
5. Strong Sun / Peak hours:
If hour between 10 and 15:
IRRADIATION range:
0.85 - 1.0
Temperature rules:
MODULE_TEMPERATURE is usually higher than ambient.
Normal:
MODULE_TEMPERATURE =
AMBIENT_TEMPERATURE + 5 to 15
Hot condition:
MODULE_TEMPERATURE can be:
AMBIENT_TEMPERATURE + 15 to 25
Return ONLY valid JSON.
Format:
{{
"AMBIENT_TEMPERATURE": {temp_val},
"MODULE_TEMPERATURE": value,
"IRRADIATION": value,
"HOUR": {time_val.hour},
"MINUTE": {time_val.minute},
"DAY_OF_YEAR": {day_of_year}
}}
"""
        response = ask_gemini(prompt)
        clean_json = (
            response
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        data = json.loads(clean_json)
        # Safety checks before sending to ML model
        if data["HOUR"] < 6 or data["HOUR"] >= 19:
            data["IRRADIATION"] = 0.0
        if data["IRRADIATION"] < 0:
            data["IRRADIATION"] = 0.0
        if data["IRRADIATION"] > 1:
            data["IRRADIATION"] = 1.0
        return data