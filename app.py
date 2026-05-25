import streamlit as st
import pandas as pd
import plotly.express as px

from prediction_version2  import predict_next_7_days


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ATM Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TITLE
# =========================================================

st.title("🏧 ATM Cash Forecasting Dashboard")

st.markdown(
    """
    Predicting next 7 days ATM cash withdrawal demand
    using Machine Learning Forecasting.
    """
)

# =========================================================
# HOME BUTTON
# =========================================================

if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False

if not st.session_state.show_prediction:

    st.markdown("## Welcome")

    st.write(
        """
        Click below to generate next 7 days ATM predictions.
        """
    )

    if st.button("🚀 Generate Next 7 Days Prediction"):

        st.session_state.show_prediction = True

    st.stop()

# =========================================================
# LOAD DATA
# =========================================================

with st.spinner("Generating Predictions..."):

    final_pred_df = predict_next_7_days()

# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.title("📌 Filters")

selected_state = st.sidebar.selectbox(
    "Select State",
    sorted(final_pred_df["State_Name"].unique())
)

state_df = final_pred_df[
    final_pred_df["State_Name"] == selected_state
]

selected_city = st.sidebar.selectbox(
    "Select City",
    sorted(state_df["ATM_City"].unique())
)

city_df = state_df[
    state_df["ATM_City"] == selected_city
]

date_columns = final_pred_df.columns[4:]

# =========================================================
# STATE LEVEL
# =========================================================

st.markdown("---")
st.header(f"📍 {selected_state} - 7 Days Forecast")

state_total = (
    state_df[date_columns]
    .sum()
    .reset_index()
)

state_total.columns = ["Date", "Total Amount"]

col1, col2 = st.columns(2)

with col1:

    if st.button("➕ State Total Sum"):

        st.success(
            f"Total Forecast Amount: ₹ {state_total['Total Amount'].sum():,.2f}"
        )

with col2:

    if st.button("📊 State Average"):

        st.info(
            f"Average Forecast Amount: ₹ {state_total['Total Amount'].mean():,.2f}"
        )

st.dataframe(
    state_total,
    use_container_width=True
)

if st.button("📈 Plot State Forecast"):

    fig = px.line(
        state_total,
        x="Date",
        y="Total Amount",
        markers=True,
        title=f"{selected_state} Forecast"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# CITY LEVEL
# =========================================================

st.markdown("---")
st.header(f"🏙️ {selected_city} - City Forecast")

city_total = (
    city_df[date_columns]
    .sum()
    .reset_index()
)

city_total.columns = ["Date", "Total Amount"]

col3, col4 = st.columns(2)

with col3:

    if st.button("➕ City Total Sum"):

        st.success(
            f"Total Forecast Amount: ₹ {city_total['Total Amount'].sum():,.2f}"
        )

with col4:

    if st.button("📊 City Average"):

        st.info(
            f"Average Forecast Amount: ₹ {city_total['Total Amount'].mean():,.2f}"
        )

st.dataframe(
    city_total,
    use_container_width=True
)

if st.button("📈 Plot City Forecast"):

    fig = px.bar(
        city_total,
        x="Date",
        y="Total Amount",
        title=f"{selected_city} Forecast"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# ATM LEVEL
# =========================================================

st.markdown("---")
st.header("🏧 ATM Level Prediction")

atm_list = city_df["ATM ID"].unique()

selected_atm = st.selectbox(
    "Select ATM",
    atm_list
)

atm_df = city_df[
    city_df["ATM ID"] == selected_atm
]

st.dataframe(
    atm_df,
    use_container_width=True
)

atm_forecast = (
    atm_df[date_columns]
    .T
    .reset_index()
)

atm_forecast.columns = ["Date", "Amount"]

col5, col6 = st.columns(2)

with col5:

    if st.button("➕ ATM Total Sum"):

        st.success(
            f"Total Forecast Amount: ₹ {atm_forecast['Amount'].sum():,.2f}"
        )

with col6:

    if st.button("📊 ATM Average"):

        st.info(
            f"Average Forecast Amount: ₹ {atm_forecast['Amount'].mean():,.2f}"
        )

if st.button("📈 Plot ATM Forecast"):

    fig = px.line(
        atm_forecast,
        x="Date",
        y="Amount",
        markers=True,
        title=f"ATM {selected_atm} Forecast"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# FULL DATAFRAME
# =========================================================

st.markdown("---")

if st.button("📂 Show Full Prediction Data"):

    st.dataframe(
        final_pred_df,
        use_container_width=True
    )

# =========================================================
# DOWNLOAD BUTTON
# =========================================================

csv = final_pred_df.to_csv(index=False)

st.download_button(
    label="⬇️ Download Prediction CSV",
    data=csv,
    file_name="atm_7_days_prediction.csv",
    mime="text/csv"
)