import streamlit as st
from datetime import datetime
import pandas as pd
from fpdf import FPDF
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
import re


from agents.input_agent import InputAgent
from agents.prediction_agent import PredictionAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.decision_agent import DecisionAgent


load_dotenv()


# ================= IST =================

IST = ZoneInfo("Asia/Kolkata")
current_time = datetime.now(IST)



# ================= PAGE =================

st.set_page_config(
    page_title="SolarWise AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)



# ================= CLEAN REPORT =================

def clean_report(text):

    text = re.sub(
        r"<[^>]*>",
        "",
        text
    )

    return text.strip()



# ================= CSS =================

st.markdown(
"""
<style>

.main-title{
font-size:42px;
font-weight:800;
background:linear-gradient(90deg,#f59e0b,#ef4444);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.subtitle{
color:#64748b;
font-size:18px;
}


.card{
background:white;
padding:25px;
border-radius:22px;
box-shadow:0 10px 30px rgba(0,0,0,0.08);
border:1px solid #e2e8f0;
}


.metric{
font-size:38px;
font-weight:800;
color:#f59e0b;
}


.small-title{
color:#64748b;
font-weight:700;
}


.report{
background:white;
padding:30px;
border-radius:20px;
border-left:8px solid #f59e0b;
box-shadow:0 10:30px rgba(0,0,0,.08);
}

</style>
""",
unsafe_allow_html=True
)



st.markdown(
"""
<div class="main-title">
☀️ SolarWise AI
</div>

<div class="subtitle">
Multi-Agent Explainable Solar AC Power Prediction System
</div>

<br>
""",
unsafe_allow_html=True
)




# ================= PDF =================


def generate_pdf(report,power,daily_yield,features):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font(
        "Helvetica",
        "B",
        20
    )

    pdf.cell(
        0,
        15,
        "SolarWise AI Report",
        ln=True,
        align="C"
    )


    pdf.set_font(
        "Helvetica",
        size=12
    )


    pdf.cell(
        0,
        10,
        f"Generated IST: {datetime.now(IST)}",
        ln=True
    )


    pdf.ln(5)


    pdf.cell(
        0,
        10,
        f"AC Power: {power:.2f} W",
        ln=True
    )


    pdf.cell(
        0,
        10,
        f"Daily Yield: {daily_yield:.2f} kWh",
        ln=True
    )


    pdf.cell(
        0,
        10,
        f"Temperature: {features['AMBIENT_TEMPERATURE']} C",
        ln=True
    )


    pdf.cell(
        0,
        10,
        f"Irradiation: {features['IRRADIATION']}",
        ln=True
    )


    pdf.ln(10)


    pdf.multi_cell(
        0,
        8,
        report.encode(
            "ascii",
            "ignore"
        ).decode()
    )


    return bytes(pdf.output())





# ================= SIDEBAR =================


with st.sidebar:

    st.header("⚙️ Solar Parameters")


    input_date = st.date_input(
        "Date",
        current_time.date()
    )


    input_time = st.time_input(
        "Current Time (IST)",
        current_time.time()
    )


    input_temp = st.slider(
        "Ambient Temperature °C",
        -10,
        60,
        30
    )


    weather = st.selectbox(
        "Weather",
        [
            "Clear Sunny Day",
            "Partly Cloudy",
            "Overcast/Cloudy",
            "Light Rain",
            "Heavy Rain/Storm"
        ]
    )


    context = st.text_area(
        "Extra Context"
    )


    run = st.button(
        "🚀 RUN AI ANALYSIS",
        use_container_width=True
    )





# ================= PIPELINE =================


if run:

    with st.spinner(
        "🤖 Agents analysing solar conditions..."
    ):

        try:


            input_agent = InputAgent()


            features = input_agent.extract(
                input_date,
                input_time,
                input_temp,
                weather,
                context
            )



            predictor = PredictionAgent()


            prediction = predictor.predict(
                features
            )


            power = prediction[
                "predicted_ac_power_watt"
            ]


            X = prediction["dataframe"]



            # DAILY YIELD

            hour = features["HOUR"]

            irradiation = features["IRRADIATION"]


            if hour < 6:
                effective_hours = 8

            elif hour >= 19:
                effective_hours = 0

            else:
                effective_hours = 18-hour



            daily_yield = (
                power *
                irradiation *
                effective_hours
            ) / 1000





            # SHAP

            explainer = ExplainabilityAgent()


            explanation = explainer.explain(
                predictor.model,
                X
            )




            decision = DecisionAgent()


            report = decision.generate(
                prediction,
                features,
                explanation
            )


            report = clean_report(report)




            # ================= DASHBOARD =================


            c1,c2,c3,c4 = st.columns(4)


            with c1:

                st.markdown(
                f"""
                <div class="card">
                <div class="small-title">
                ⚡ AC POWER
                </div>

                <div class="metric">
                {power:.1f}
                </div>

                Watt
                </div>
                """,
                unsafe_allow_html=True
                )



            with c2:

                st.markdown(
                f"""
                <div class="card">
                <div class="small-title">
                🔋 DAILY YIELD
                </div>

                <div class="metric">
                {daily_yield:.2f}
                </div>

                kWh
                </div>
                """,
                unsafe_allow_html=True
                )



            with c3:

                st.markdown(
                f"""
                <div class="card">
                <div class="small-title">
                🌡 TEMP
                </div>

                <div class="metric">
                {features['AMBIENT_TEMPERATURE']}
                </div>
                °C
                </div>
                """,
                unsafe_allow_html=True
                )



            with c4:


                if hour < 6 or hour >=19:
                    status="🌙 Night"

                elif 10<=hour<=15:
                    status="☀️ Peak"

                else:
                    status="🌤 Normal"



                st.markdown(
                f"""
                <div class="card">
                <div class="small-title">
                STATUS
                </div>

                <div class="metric">
                {status}
                </div>

                </div>
                """,
                unsafe_allow_html=True
                )





            st.divider()



            left,right = st.columns(2)



            with left:

                st.subheader(
                    "📊 Feature Impact"
                )


                shap_data = explanation.get(
                    "shap_values",
                    {}
                )


                df = pd.DataFrame(
                    list(shap_data.items()),
                    columns=[
                        "Feature",
                        "Impact"
                    ]
                )


                st.bar_chart(
                    df.set_index("Feature")
                )





            with right:

                st.subheader(
                    "🔍 AI Reasoning"
                )


                for x in explanation.get(
                    "analysis",
                    []
                ):

                    st.write(
                        "✔️",
                        x
                    )





            st.divider()



            st.subheader(
                "🤖 Expert Solar Report"
            )


            st.markdown(
                report
            )



            # SYSTEM NOTICE

            st.markdown(
            """
            <div class="card">

            <h3>🛠️ System Reliability Notice</h3>

            <p>
            This prediction is generated using AI based solar analysis.
            If production is lower than expected, inspect:
            </p>

            <ul>
            <li>☀️ Panel dust or shading</li>
            <li>🔌 Inverter performance</li>
            <li>⚡ Wiring losses</li>
            <li>🌡 Temperature losses</li>
            <li>🔧 Hardware maintenance</li>
            </ul>

            <p>
            For abnormal values, consult a qualified solar technician.
            </p>

            </div>
            """,
            unsafe_allow_html=True
            )



            pdf = generate_pdf(
                report,
                power,
                daily_yield,
                features
            )



            st.download_button(
                "📥 Download Professional Report",
                pdf,
                "SolarWise_Report.pdf",
                "application/pdf",
                use_container_width=True
            )



        except Exception as e:

            st.error(
                f"Pipeline Error: {e}"
            )
