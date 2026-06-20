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

        hour = features["HOUR"]
        minute = features["MINUTE"]

        irradiation = features["IRRADIATION"]

        ambient_temp = features["AMBIENT_TEMPERATURE"]
        module_temp = features["MODULE_TEMPERATURE"]


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
        # Daily Yield
        # ----------------------------

        power_kw = ac_power / 1000


        estimated_yield = (
            power_kw *
            irradiation *
            effective_hours
        )


        estimated_yield = round(
            estimated_yield,
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





        prompt = f"""

You are an expert solar plant consultant.

Create a professional solar performance report.

DATA:

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


Feature Impact:
{explanation}


Create report:

## 📊 Executive Summary

Explain current solar plant status.


## ⚡ Power Generation

Explain whether current AC power is normal.


## 🔋 Expected Daily Yield

Explain expected energy production.


## 🔍 Why This Prediction?

Explain:

- irradiation effect
- temperature effect
- time effect


## ⚠️ Performance Check

Mention:

- night condition
- low irradiation
- temperature losses


## 🛠 Hardware Recommendation

If output looks abnormal suggest:

- inverter checking
- panel cleaning
- wiring inspection
- technician inspection if required


## ✅ Recommendations

Give 3 practical suggestions.


Rules:

- Use Markdown only
- Do not use HTML
- Never write <div>
- Keep professional language

"""



        # ----------------------------
        # Gemini + fallback
        # ----------------------------

        try:

            response = ask_gemini(
                prompt
            )


            response = (
                response
                .replace("<div>", "")
                .replace("</div>", "")
                .replace("```html", "")
                .replace("```", "")
            )


            return response



        except Exception:


            # Local fallback report

            return f"""

## 📊 Executive Summary

SolarWise AI predicted the current solar plant condition.

Current AC Power:

**{ac_power:.2f} Watt**


Solar Period:

**{period}**



## ⚡ Power Generation

The predicted generation depends on:

- Solar irradiation
- Temperature
- Current time


Current condition:

**{weather}**



## 🔋 Expected Daily Yield

Estimated daily production:

**{estimated_yield} kWh**



## 🔍 Why This Prediction?

### ☀️ Irradiation

Irradiation value:

**{irradiation}**

Higher irradiation improves generation.



### 🌡 Temperature

Module temperature:

**{module_temp} °C**

High temperature can reduce efficiency.



### 🕒 Time

Current solar time:

**{current_time}**



## ⚠️ Performance Check


"""


            if hour < 6 or hour >= 19:

                return "Night condition detected. Solar generation is expected to be zero."

            elif irradiation < 0.2:

                return "Low sunlight detected. Reduced output expected."

            else:

                return """
Solar generation appears normal.


## 🛠 Hardware Recommendation

If actual plant output differs from prediction:

- Check inverter status
- Inspect panel cleanliness
- Verify cable connections
- Contact solar technician for hardware inspection


## ✅ Recommendations

1. Maintain clean solar panels.
2. Monitor inverter health.
3. Compare predicted vs actual generation regularly.
"""
