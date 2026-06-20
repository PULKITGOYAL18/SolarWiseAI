from agents.gemini_core import ask_gemini


class DecisionAgent:

    def generate(
        self,
        prediction,
        features,
        explanation
    ):

        # ----------------------------
        # Values
        # ----------------------------

        ac_power = prediction.get(
            "predicted_ac_power_watt",
            0
        )

        hour = int(features["HOUR"])
        minute = int(features["MINUTE"])

        irradiation = float(
            features["IRRADIATION"]
        )

        ambient_temp = float(
            features["AMBIENT_TEMPERATURE"]
        )

        module_temp = float(
            features["MODULE_TEMPERATURE"]
        )


        current_time = f"{hour:02d}:{minute:02d}"


        # ----------------------------
        # Solar Period
        # ----------------------------

        if hour < 6 or hour >= 19:

            period = "Night"
            effective_hours = 0

        elif 10 <= hour <= 15:

            period = "Peak Sun Hours"
            effective_hours = 5

        elif 6 <= hour < 10:

            period = "Morning"
            effective_hours = 6

        else:

            period = "Evening"
            effective_hours = 3



        # ----------------------------
        # Yield
        # ----------------------------

        power_kw = ac_power / 1000


        estimated_yield = round(
            power_kw *
            irradiation *
            effective_hours,
            2
        )



        # ----------------------------
        # Weather
        # ----------------------------

        if irradiation < 0.2:

            weather = "Low sunlight condition"

        elif irradiation < 0.7:

            weather = "Moderate sunlight condition"

        else:

            weather = "Good solar condition"



        # ----------------------------
        # Gemini Prompt
        # ----------------------------

        prompt = f"""

You are a solar plant expert.

Generate a professional report.

Current Time:
{current_time}

Solar Period:
{period}

AC Power:
{ac_power:.2f} Watt


Estimated Daily Yield:
{estimated_yield} kWh


Weather:
{weather}


Ambient Temperature:
{ambient_temp} °C


Module Temperature:
{module_temp} °C


Explainable AI Feature Impact:
{explanation}


Report format:

## 📊 Executive Summary

## ⚡ Power Generation Status

## 🔋 Expected Daily Yield

## 🔍 Why This Prediction?

Explain:
- irradiation effect
- temperature effect
- time effect


## ⚠️ Performance Check

Mention:
- night condition
- low sunlight
- temperature losses


## 🛠 Hardware Recommendation

If prediction and actual output differ:

- Check inverter
- Inspect wiring
- Clean solar panels
- Contact technician if required


## ✅ Recommendations


Rules:
- Markdown only
- No HTML
- Do not write div tags

"""


        try:

            response = ask_gemini(
                prompt
            )


            if response is None:

                raise Exception(
                    "Empty Gemini response"
                )


            response = str(response)


            response = (
                response
                .replace("<div>", "")
                .replace("</div>", "")
                .replace("```html", "")
                .replace("```", "")
            )


            return response



        except Exception:


            # ----------------------------
            # Local fallback
            # ----------------------------

            if hour < 6 or hour >= 19:

                status = (
                    "Night condition detected. "
                    "Solar generation is expected near zero."
                )

            elif irradiation < 0.2:

                status = (
                    "Low irradiation detected. "
                    "Reduced generation expected."
                )

            else:

                status = (
                    "Solar generation condition appears normal."
                )


            return f"""

## 📊 Executive Summary

SolarWise AI completed the solar analysis.


## ⚡ Power Generation Status


Predicted AC Power:

**{ac_power:.2f} Watt**


Condition:

**{status}**


## 🔋 Expected Daily Yield


Estimated yield:

**{estimated_yield} kWh**



## 🔍 Why This Prediction?


☀️ Irradiation:

{irradiation}


🌡 Module Temperature:

{module_temp} °C


🕒 Current Time:

{current_time}



## 🛠 Hardware Recommendation


If real output differs from prediction:

- Check inverter status
- Inspect panel cleanliness
- Verify cable connections
- Contact solar technician


## ✅ Recommendations


1. Maintain clean panels.
2. Monitor inverter performance.
3. Compare predicted and actual power regularly.

"""
