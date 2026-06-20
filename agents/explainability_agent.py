import shap
import pandas as pd
import os


class ExplainabilityAgent:

    def __init__(self):

        # Works locally + Render
        self.background_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "solar_data4.csv"
        )


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


            # ensure dataframe
            X = X.copy()

            X = X[order]


            # load background data
            full_df = pd.read_csv(
                self.background_path
            )


            background = full_df[order]


            # reduce calculation
            background_sample = shap.sample(
                background,
                20,
                random_state=42
            )


            # Tree models support TreeExplainer
            explainer = shap.TreeExplainer(
                model
            )


            shap_values = explainer.shap_values(
                X
            )


            if isinstance(shap_values, list):
                values = shap_values[0]
            else:
                values = shap_values



            shap_result = {}


            for feature, value in zip(
                order,
                values[0]
            ):

                shap_result[feature] = round(
                    float(value),
                    4
                )



            irradiation = float(
                X["IRRADIATION"].iloc[0]
            )

            ambient = float(
                X["AMBIENT_TEMPERATURE"].iloc[0]
            )

            module = float(
                X["MODULE_TEMPERATURE"].iloc[0]
            )

            hour = int(
                X["HOUR"].iloc[0]
            )


            reasons=[]


            if hour < 6 or hour >= 19:

                reasons.append(
                    "🌙 Night detected. Solar generation is naturally near zero."
                )


            elif irradiation > 0.7:

                reasons.append(
                    "☀️ High irradiation is increasing power generation."
                )


            else:

                reasons.append(
                    "☁️ Lower irradiation is limiting generation."
                )



            if module > ambient + 15:

                reasons.append(
                    "🌡 High module temperature can reduce efficiency."
                )

            else:

                reasons.append(
                    "🌡 Temperature condition is normal."
                )



            return {

                "shap_values": shap_result,

                "analysis": reasons

            }



        except Exception as e:

            print(
                "SHAP ERROR:",
                e
            )

            return {

                "shap_values": {},

                "analysis":[
                    f"Explainability failed: {e}"
                ]

            }
