import joblib
import pandas as pd
class PredictionAgent:
    def __init__(self):
        self.model = joblib.load(
            "model/solar_model4.pkl"
        )
        # exact training order
        self.order = [
            "HOUR",
            "MINUTE",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION",
            "DAY_OF_YEAR"
        ]
    def predict(self, features):
        X = pd.DataFrame(
            [[
                features["HOUR"],
                features["MINUTE"],
                features["AMBIENT_TEMPERATURE"],
                features["MODULE_TEMPERATURE"],
                features["IRRADIATION"],
                features["DAY_OF_YEAR"]
            ]],
            columns=self.order
        )
        # ensure correct order
        X = X[self.order]
        prediction = self.model.predict(
            X
        )[0]
        # night protection
        if features["HOUR"] < 6 or features["HOUR"] >= 19:
            prediction = 0
        return {
            "predicted_ac_power_watt": float(
                prediction
            ),
            # required for SHAP
            "dataframe": X,
            "input_features": features,
            "unit": "W"
        }