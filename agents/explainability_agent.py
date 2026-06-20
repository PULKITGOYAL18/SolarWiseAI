import shap
import pandas as pd
import os
import traceback


class ExplainabilityAgent:

    def __init__(self):

        # Works on local + Render
        self.background_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "solar_data4.csv"
        )


    def explain(self, model, X):

        try:

            # Exact training feature order
            order = [
                "HOUR",
                "MINUTE",
                "AMBIENT_TEMPERATURE",
                "MODULE_TEMPERATURE",
                "IRRADIATION",
                "DAY_OF_YEAR"
            ]


            # Ensure dataframe
            X = X.copy()

            X = X[order]


            # Load background dataset
            background_df = pd.read_csv(
                self.background_path
            )


            background = background_df[order]


            # small sample for Render memory
            background_sample = background.sample(
                n=20,
                random_state=42
            )


            # Pipeline safe SHAP
            explainer = shap.Explainer(
                model.predict,
                background_sample
            )


            shap_values = explainer(
                X
            )


            values = shap_values.values[0]


            shap_result = {}


            for feature, value in zip(
                order,
                values
            ):

                shap_result[feature] = round(
                    float(value),
                    4
                )


            # ---------------------------
            # Human Explanation
            # ---------------------------

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
                    "🌙 Night condition detected. Solar generation is naturally near zero."
                )


            elif irradiation > 0.75:

                reasons.append(
                    "☀️ High irradiation is the main contributor increasing power output."
                )


            elif irradiation < 0.2:

                reasons.append(
                    "☁️ Low irradiation is reducing solar generation."
                )


            else:

                reasons.append(
                    "🌤 Moderate sunlight condition is affecting production."
                )



            if module_temp > ambient + 15:

                reasons.append(
                    "🌡 Higher module temperature can reduce panel efficiency."
                )

            else:

                reasons.append(
                    "🌡 Temperature conditions are within normal operating range."
                )



            if 10 <= hour <= 15:

                reasons.append(
                    "⚡ Peak solar hours detected."
                )



            return {

                "shap_values": shap_result,

                "analysis": reasons

            }



        except Exception as e:


            print(
                "========== SHAP ERROR =========="
            )

            print(
                str(e)
            )

            traceback.print_exc()

            print(
                "================================"
            )


            return {

                "shap_values": {},

                "analysis": [
                    f"SHAP Error: {str(e)}"
                ]

            }
