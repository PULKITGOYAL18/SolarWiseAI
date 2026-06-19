import shap
import pandas as pd
import numpy as np
class ExplainabilityAgent:
    def __init__(self):
        # dataset is used only for background sampling
        self.background_path = "solar_data4.csv"
        self.order = [
            "HOUR",
            "MINUTE",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION",
            "DAY_OF_YEAR"
        ]
    def explain(self, model, X):
        try:
            # -----------------------------
            # STEP 1: Ensure correct feature order
            # -----------------------------
            X = X[self.order]
            # -----------------------------
            # STEP 2: Load background dataset
            # -----------------------------
            full_df = pd.read_csv(self.background_path)
            background = full_df[self.order]
            # small sample = faster + cloud safe
            background_sample = shap.sample(background, 20)
            # -----------------------------
            # STEP 3: FORCE XGBOOST TREE EXPLAINER
            # -----------------------------
            # This is FAST + STABLE for XGBoost/Tree models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            # If binary classification / regression
            if isinstance(shap_values, list):
                values = shap_values[0]
            else:
                values = shap_values
            # -----------------------------
            # STEP 4: Convert SHAP values
            # -----------------------------
            real_values = {
                feature: round(float(val), 4)
                for feature, val in zip(self.order, values[0])
            }
            # -----------------------------
            # STEP 5: Human explanation logic
            # -----------------------------
            irradiation = float(X["IRRADIATION"].iloc[0])
            ambient = float(X["AMBIENT_TEMPERATURE"].iloc[0])
            module_temp = float(X["MODULE_TEMPERATURE"].iloc[0])
            hour = int(X["HOUR"].iloc[0])
            reasons = []
            if hour < 6 or hour >= 19:
                reasons.append("🌙 Night condition detected. No solar generation expected.")
            elif irradiation > 0.75:
                reasons.append("☀️ High irradiation is driving strong power generation.")
            elif irradiation < 0.2:
                reasons.append("☁️ Low irradiation is reducing solar output.")
            else:
                reasons.append("🌤 Moderate sunlight producing normal generation.")
            if module_temp > ambient + 15:
                reasons.append("🌡 High module temperature slightly reduces efficiency.")
            else:
                reasons.append("🌡 Temperature is within optimal range.")
            if 10 <= hour <= 15:
                reasons.append("⚡ Peak solar hours improving output.")
            else:
                reasons.append("🕒 Non-peak hours affecting generation.")
            # -----------------------------
            # FINAL OUTPUT
            # -----------------------------
            return {
                "real_shap_values": real_values,
                "analysis": reasons
            }
        except Exception as e:
            print("SHAP ERROR:", str(e))
            return {
                "real_shap_values": {},
        analysis": ["Explainability unavailable"]
            }
