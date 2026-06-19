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
            weather = (
                "Low sunlight condition"
            )
        elif irradiation < 0.7:
            weather = (
                "Moderate sunlight condition"
            )
        else:
            weather = (
                "Good solar condition"
            )
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
Write report with:
## 📊 Executive Summary
Explain current solar plant status.
## ⚡ Power Generation
Explain whether {ac_power:.2f} Watt AC power is normal
for the given sunlight condition.
## 🔋 Expected Daily Yield
Explain the estimated {estimated_yield} kWh yield.
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
## ✅ Recommendations
Give 3 practical suggestions.
Rules:

- Use Markdown
- Do not use HTML
- Do not write <div> or </div>
- Keep language simple and professional
"""
        response = ask_gemini(prompt)
        # Remove accidental HTML
        response = (
            response
            .replace("<div>", "")
            .replace("</div>", "")
            .replace("```html", "")
            .replace("```", "")
        )
        return response