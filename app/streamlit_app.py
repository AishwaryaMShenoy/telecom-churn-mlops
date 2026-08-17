import requests
import pandas as pd
import streamlit as st


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Telecom Customer Intelligence",
    page_icon="📱",
    layout="wide"
)


# ==================================================
# THEME TOGGLE
# ==================================================

theme_col, toggle_col = st.columns([8, 2])

with toggle_col:

    dark_mode = st.toggle(
        "Dark mode",
        value=False
    )


# ==================================================
# THEME COLORS
# ==================================================

if dark_mode:

    background = "#0e1117"
    card = "#1b2430"
    text = "#f5f7fa"
    secondary = "#b8c4d0"
    border = "#394858"
    header = "#172a46"

else:

    background = "#f4f6f9"
    card = "#ffffff"
    text = "#102a43"
    secondary = "#52606d"
    border = "#d9e2ec"
    header = "#102a43"


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    f"""
    <style>

    /* ----------------------------------------------
       Application background
    ---------------------------------------------- */

    .stApp {{
        background-color: {background};
    }}


    /* ----------------------------------------------
       Main header
    ---------------------------------------------- */

    .main-header {{
        background-color: {header};
        padding: 25px 32px;
        border-radius: 10px;
        margin-bottom: 25px;
    }}

    .main-header h1 {{
        color: white;
        margin: 0;
        font-size: 30px;
        font-weight: 600;
    }}

    .main-header p {{
        color: #d9e2ec;
        margin: 6px 0 0 0;
        font-size: 15px;
    }}


    /* ----------------------------------------------
       Section titles
    ---------------------------------------------- */

    .section-title {{
        color: {text};
        font-size: 20px;
        font-weight: 600;
        margin-top: 22px;
        margin-bottom: 12px;
    }}


    /* ----------------------------------------------
       Theme toggle visibility
    ---------------------------------------------- */

    div[data-testid="stCheckbox"] {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 8px;
        padding: 8px 12px;
    }}

    div[data-testid="stCheckbox"] label {{
        color: {text} !important;
    }}


    /* ----------------------------------------------
       Search input
    ---------------------------------------------- */

    div[data-testid="stTextInput"] input {{
        background-color: {card};
        color: {text};
        border: 1px solid {border};
        border-radius: 7px;
    }}


    /* ----------------------------------------------
       Buttons
    ---------------------------------------------- */

    div.stButton > button {{
        border-radius: 7px;
        font-weight: 600;
        height: 42px;
    }}


    /* ----------------------------------------------
       Metric cards
    ---------------------------------------------- */

    div[data-testid="metric-container"] {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 8px;
        padding: 14px;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {secondary};
    }}


    /* ----------------------------------------------
       Risk cards
    ---------------------------------------------- */

    .risk-high {{
        background-color: {"#3b2022" if dark_mode else "#fff1f0"};
        border-left: 6px solid #d64545;
        padding: 18px;
        border-radius: 8px;
    }}

    .risk-medium {{
        background-color: {"#3d321d" if dark_mode else "#fff8e6"};
        border-left: 6px solid #d99a00;
        padding: 18px;
        border-radius: 8px;
    }}

    .risk-low {{
        background-color: {"#19352a" if dark_mode else "#edf8f2"};
        border-left: 6px solid #2f855a;
        padding: 18px;
        border-radius: 8px;
    }}

    .risk-title {{
        color: {"white" if dark_mode else "#102a43"};
        font-size: 21px;
        font-weight: 700;
    }}

    .risk-text {{
        color: {"#d9e2ec" if dark_mode else "#52606d"};
        font-size: 14px;
        margin-top: 5px;
    }}


    /* ----------------------------------------------
       Recommendation card
    ---------------------------------------------- */

    .recommendation {{
        background-color: {card};
        border: 1px solid {border};
        border-left: 6px solid #2f6fed;
        padding: 18px;
        border-radius: 8px;
        color: {text};
    }}


    /* ----------------------------------------------
       Footer
    ---------------------------------------------- */

    .footer {{
        text-align: center;
        color: {secondary};
        font-size: 12px;
        margin-top: 35px;
        padding-bottom: 15px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
    <div class="main-header">
        <h1>Telecom Customer Intelligence</h1>
        <p>
            Customer-service decision support and churn risk assessment
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# CUSTOMER LOOKUP
# ==================================================

st.markdown(
    '<div class="section-title">Customer Lookup</div>',
    unsafe_allow_html=True
)

search_col, button_col = st.columns([5, 1])

with search_col:

    phone_number = st.text_input(
        "Registered mobile number",
        placeholder="Enter customer mobile number"
    )

with button_col:

    st.write("")

    search = st.button(
        "Search Customer",
        use_container_width=True
    )


# ==================================================
# SEARCH
# ==================================================

if search:

    phone_number = phone_number.strip()

    if not phone_number:

        st.warning(
            "Please enter a customer mobile number."
        )

    else:

        try:

            response = requests.post(
                "http://api:8000/customer",
                json={
                    "phone_number": phone_number
                },
                timeout=10
            )

            if response.status_code == 404:

                st.error(
                    "No customer record was found for this mobile number."
                )

            else:

                response.raise_for_status()

                result = response.json()

                profile = result["customer_profile"]

                trends = result["monthly_trends"]

                assessment = result["churn_assessment"]

                recommendation = result[
                    "recommended_action"
                ]


                # ==================================================
                # CUSTOMER PROFILE
                # ==================================================

                st.markdown(
                    '<div class="section-title">'
                    'Customer Profile'
                    '</div>',
                    unsafe_allow_html=True
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Customer ID",
                        profile["customer_id"]
                    )

                with col2:

                    st.metric(
                        "Tenure",
                        f"{profile['tenure_months']} months"
                    )

                with col3:

                    st.metric(
                        "Average ARPU",
                        f"₹{profile['average_arpu']:.2f}"
                    )

                with col4:

                    st.metric(
                        "Average Recharge",
                        f"₹{profile['average_recharge']:.2f}"
                    )


                # ==================================================
                # CHURN ASSESSMENT
                # ==================================================

                st.markdown(
                    '<div class="section-title">'
                    'Churn Risk Assessment'
                    '</div>',
                    unsafe_allow_html=True
                )

                probability = assessment["probability"]

                risk = assessment["risk_level"]


                if risk == "HIGH":

                    risk_class = "risk-high"

                    risk_title = "HIGH CHURN RISK"

                elif risk == "MEDIUM":

                    risk_class = "risk-medium"

                    risk_title = "MEDIUM CHURN RISK"

                else:

                    risk_class = "risk-low"

                    risk_title = "LOW CHURN RISK"


                # Simple HTML only — no nested HTML blocks
                st.markdown(
                    f"""
                    <div class="{risk_class}">
                        <div class="risk-title">
                            {risk_title}
                        </div>
                        <div class="risk-text">
                            Estimated churn probability:
                            <strong>
                                {probability * 100:.2f}%
                            </strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # ==================================================
                # RECOMMENDED ACTION
                # ==================================================

                st.markdown(
                    '<div class="section-title">'
                    'Recommended Agent Action'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="recommendation">
                        <strong>Recommended response</strong>
                        <br><br>
                        {recommendation}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # ==================================================
                # CUSTOMER BEHAVIOUR
                # ==================================================

                st.markdown(
                    '<div class="section-title">'
                    'Customer Behaviour'
                    '</div>',
                    unsafe_allow_html=True
                )


                # Convert API response into a DataFrame
                behaviour_data = []

                for month in ["6", "7", "8"]:

                    data = trends[month]

                    behaviour_data.append({

                        "Period": f"Month {month}",

                        "ARPU": round(
                            data["arpu"],
                            2
                        ),

                        "Recharge": round(
                            data["recharge"],
                            2
                        ),

                        "Outgoing Usage": round(
                            data["outgoing_usage"],
                            2
                        ),

                        "Incoming Usage": round(
                            data["incoming_usage"],
                            2
                        ),

                        "2G Usage": round(
                            data["2g_usage"],
                            2
                        ),

                        "3G Usage": round(
                            data["3g_usage"],
                            2
                        )
                    })


                behaviour_df = pd.DataFrame(
                    behaviour_data
                )


                st.dataframe(
                    behaviour_df,
                    use_container_width=True,
                    hide_index=True
                )


        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the FastAPI server. "
                "Make sure FastAPI is running on port 8000."
            )

        except requests.exceptions.RequestException as error:

            st.error(
                f"API request failed: {error}"
            )


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">
        Telecom Customer Intelligence • ML-powered decision support
    </div>
    """,
    unsafe_allow_html=True
)