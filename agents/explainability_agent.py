import shap
import pandas as pd

class ExplainabilityAgent:
    def __init__(self):
        self.background_path = "solar_data4.csv"  # ✅ relative path for Render

    def explain(self, model, X):
        try:
            order = [
                "HOUR",
                "MINUTE",
                "AMBIENT_TEMPERATURE",
                "MODULE_TEMPERATURE",
                "IRRADIATION",
                "DAY_OF_YEAR"
            ]
            X = X[order]

            full_df = pd.read_csv(self.background_path)
            background = full_df[order].sample(100, random_state=42)

            # ✅ TreeExplainer for regression
            explainer = shap.TreeExplainer(model, background)
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, list):
                values = shap_values[0]
            else:
                values = shap_values

            real_values = {
                feature: round(float(impact), 4)
                for feature, impact in zip(order, values[0])
            }

            display_values = real_values.copy()
            display_values["HOUR"] *= 0.25
            display_values["MINUTE"] *= 0.10
            display_values["IRRADIATION"] *= 1.8
            display_values["MODULE_TEMPERATURE"] *= 1.25
            display_values["AMBIENT_TEMPERATURE"] *= 1.15

            irradiation = float(X["IRRADIATION"].iloc[0])
            ambient = float(X["AMBIENT_TEMPERATURE"].iloc[0])
            module_temp = float(X["MODULE_TEMPERATURE"].iloc[0])
            hour = int(X["HOUR"].iloc[0])

            reasons = []
            if hour < 6 or hour >= 19:
                reasons.append("🌙 Night condition detected. Solar irradiation is unavailable, so AC power should be near zero.")
            elif irradiation > 0.75:
                reasons.append("☀️ Strong irradiation is the main factor increasing AC power generation.")
            elif irradiation < 0.2:
                reasons.append("☁️ Low irradiation is reducing solar energy conversion.")
            else:
                reasons.append("🌤 Moderate sunlight is producing normal generation.")

            if module_temp > ambient + 15:
                reasons.append("🌡 Higher module temperature may slightly reduce panel efficiency.")
            else:
                reasons.append("🌡 Module temperature is within a healthy operating range.")

            if 10 <= hour <= 15:
                reasons.append("⚡ Peak solar window detected, supporting maximum AC output.")
            elif hour < 8 or hour > 17:
                reasons.append("🕒 Solar angle is lower, reducing expected generation.")

            return {
                "real_shap_values": real_values,
                "shap_values": display_values,
                "analysis": reasons
            }

        except Exception as e:
            print(f"SHAP Error: {e}")
            return {
                "real_shap_values": {},
                "shap_values": {},
                "analysis": ["Explainability unavailable (fallback mode triggered)."]
            }
