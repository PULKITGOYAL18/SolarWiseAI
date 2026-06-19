from agents.gemini_core import ask_gemini
class CoordinatorAgent:
    def run(self, user_query):
        prompt = f"""
You are a Coordinator Agent for a Solar Energy AI System.
Available agents:
1. input_agent
Role:
- Collect solar parameters
- Validate inputs
Inputs:
- AMBIENT_TEMPERATURE
- MODULE_TEMPERATURE
- IRRADIATION
- HOUR
- MINUTE
- DAY_OF_YEAR
2. prediction_agent
Role:
- Load trained ML model
- Predict AC Power output in Watts
Important:
The model predicts AC Power only.
It does NOT predict daily yield.
3. explainability_agent
Role:
Explain prediction using:
- irradiation impact
- ambient temperature
- module temperature
- time of day
Also explain:
- day/night condition
- peak generation period
4. decision_agent
Role:
Generate final solar analysis:
- estimated daily yield from AC Power
- performance status
- weather impact
- recommendations
Daily Yield calculation:
AC Power(W) → kW
Daily Yield(kWh)
=
AC Power(kW) × effective sunlight hours
5. memory_agent
Role:
- Store previous predictions
- Compare performance history
- Track improvement/degradation
User query:
{user_query}
Return ONLY JSON.
Format:
{{
"agents":[
"input_agent",
"prediction_agent",
"explainability_agent",
"decision_agent",
"memory_agent"
]
}}
Rules:
Prediction request:
input_agent + prediction_agent + decision_agent
"Why/how" request:
add explainability_agent
History/comparison request:
add memory_agent
Always include decision_agent for final report.
"""
        return ask_gemini(prompt)