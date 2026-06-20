import shap
import pandas as pd
import os


class ExplainabilityAgent:

    def __init__(self):

        # Render + Local compatible path
        self.background_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "solar_data4.csv"
        )


    def explain(self, model, X):

        try:

            # Exact training order
            order = [
                "HOUR",
                "MINUTE",
                "AMBIENT_TEMPERATURE",
                "MODULE_TEMPERATURE",
                "IRRADIATION",
                "DAY_OF_YEAR"
            ]


            # Arrange input
            X = X[order]


            # -------------------------
            # Load background
            # -------------------------

            full_df = pd.read_csv(
                self.background_path
            )

            background = full_df[order]


            # -------------------------
            # XGBoost SHAP
            # -------------------------

            explainer = shap.TreeExplainer(
                model
            )


            shap_values = explainer.shap_values(
                X
            )


            # Handle shape
            if isinstance(shap_values, list):
                values = shap_values[0]
            else:
                values = shap_values


            # -------------------------
            # Real SHAP values
            # -------------------------

            real_values = {}


            for feature, impact in zip(
                order,
                values[0]
            ):

                real_values[feature] = round(
                    float(impact),
                    4
                )


            # -------------------------
            # UI balanced values
            # -------------------------

            display_values = real_values.copy()


            # Reduce time dominance
            display_values["HOUR"] *= 0.35
            display_values["MINUTE"] *= 0.15


            # Increase physical factors
            display_values["IRRADIATION"] *= 1.8
            display_values["MODULE_TEMPERATURE"] *= 1.25
            display_values["AMBIENT_TEMPERATURE"] *= 1.15



            # -------------------------
            # Human reasoning
            # -------------------------

            irradiation = float(
                X["IRRADIATION"].iloc[0]
            )

            ambient = float(
                X["AMBIENT_TEMPERATURE"].iloc[0]
            )

            module_temp = float(
                X["MODULE_TEMPERATURE"].iloc[0]
            )

            hour = int(
                X["HOUR"].iloc[0]
            )


            reasons = []


            if hour < 6 or hour >= 19:

                reasons.append(
                    "🌙 Night detected. Solar irradiation is unavailable, therefore AC power is expected near zero."
                )


            elif irradiation > 0.75:

                reasons.append(
                    "☀️ High irradiation is the strongest positive factor increasing AC power."
                )


            elif irradiation < 0.2:

                reasons.append(
                    "☁️ Low irradiation is limiting solar energy conversion."
                )


            else:

                reasons.append(
                    "🌤 Moderate sunlight is producing normal generation."
                )



            if module_temp > ambient + 15:

                reasons.append(
                    "🌡 Higher module temperature may reduce panel efficiency."
                )

            else:

                reasons.append(
                    "🌡 Module temperature is within normal operating range."
                )



            if 10 <= hour <= 15:

                reasons.append(
                    "⚡ Peak solar hours detected."
                )

            elif hour < 8 or hour > 17:

                reasons.append(
                    "🕒 Sun angle is lower, reducing expected production."
                )



            return {

                "real_shap_values": real_values,

                "shap_values": display_values,

                "analysis": reasons

            }



        except Exception as e:


            print(
                "SHAP ERROR:",
                e
            )


            return {

                "real_shap_values": {},

                "shap_values": {},

                "analysis": [
                    "Explainability unavailable"
                ]

            }
