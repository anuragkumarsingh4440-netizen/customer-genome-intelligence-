import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# This block configures the Streamlit application layout
# Sets page title used in browser tab
# Enables wide layout for dashboard-style UI
# Sidebar is expanded by default
# This runs once when app starts
# Core visual foundation of the app
st.set_page_config(
    page_title="Customer Genome Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# This block injects global CSS styling
# Controls background color and typography
# Defines hero heading visual appearance
# Improves executive readability
# Applies glow and gradient effects
# Affects entire application UI
st.markdown("""
<style>
body {
    background-color: #070b16;
    color: #e5e7eb;
}
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# This block renders the main hero heading
# Designed to look like a product banner
# Full-width container with gradient background
# Large font size for strong visual impact
# Matches enterprise SaaS landing style
# Only visual change requested by user
st.markdown("""
<div style="
    width:100%;
    margin: 20px auto 40px auto;
    padding: 50px 20px;
    text-align:center;
    background: linear-gradient(90deg, #1e3a8a, #6d28d9);
    border-radius: 22px;
">
    <div style="
        font-size:88px;
        font-weight:900;
        color:white;
        letter-spacing:1px;
        text-shadow: 0 0 40px rgba(255,255,255,0.35);
    ">
        🧬 Customer Genome Intelligence
    </div>
    <div style="
        margin-top:16px;
        font-size:26px;
        color:#e5e7eb;
        font-weight:500;
    ">
        Behavioral Segmentation • Risk Intelligence • Real-Time Action Strategy
    </div>
</div>
""", unsafe_allow_html=True)

# This block explains the dataset requirements
# Helps users upload correct CSV structure
# Prevents schema-related runtime errors
# Displayed before any processing begins
# Purely informational section
# No effect on model logic
st.markdown("""
### 📘 Dataset Schema (Required Columns)

- **customer_id**
- **total_orders**
- **total_quantity**
- **total_spend**
- **avg_order_value**
- **recency_days**
- **unique_products**
""")

# This block loads trained ML models
# Uses Streamlit resource cache
# Prevents repeated disk reads
# Loads clustering and prediction models
# Runs only once per session
# Improves performance
@st.cache_resource
def load_models():
    behavior_model = joblib.load("models/behavior_model.pkl")
    value_model = joblib.load("models/value_model.pkl")
    stability_model = joblib.load("models/stability_model.pkl")
    return behavior_model, value_model, stability_model

behavior_model, value_model, stability_model = load_models()

# This block defines cluster labels
# Maps numeric clusters to business meaning
# Used across dashboard views
# Improves interpretability
# Keeps logic centralized
# No computation here
CLUSTER_LABELS = {
    0: "🟢 Loyal & High Value",
    1: "🔵 Growing Customers",
    2: "🟡 Price Sensitive",
    3: "🟠 At Risk",
    4: "🔴 High Churn Risk"
}

# This block defines recommended actions
# Strategy tied to each cluster
# Used in individual intelligence view
# Business-facing explanations
# Static configuration only
# No model dependency
CLUSTER_ACTIONS = {
    0: "Retention focus, loyalty rewards, premium upsells",
    1: "Cross-sell, personalized bundles, growth nudges",
    2: "Discounts, pricing optimization, value packs",
    3: "Re-engagement campaigns, reminders, incentives",
    4: "Immediate retention calls and win-back offers"
}

# This block builds sidebar UI
# Handles CSV upload
# Explicit build button prevents auto rerun
# Keeps user control clear
# Minimal interaction logic
# No data processing here
st.sidebar.title("📂 Input Panel")
uploaded_file = st.sidebar.file_uploader("Upload Customer-Level CSV", type=["csv"])
build_button = st.sidebar.button("🚀 Build Intelligence")

# This block handles data preprocessing
# Runs only after button click
# Normalizes column names
# Scales numerical features
# Applies clustering model
# Stores results in session state
if build_button and uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    df = df.rename(columns={"customerid": "customer_id"})

    feature_cols = [
        "total_orders",
        "total_quantity",
        "total_spend",
        "avg_order_value",
        "recency_days",
        "unique_products"
    ]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    df["cluster"] = behavior_model.predict(scaled_features)

    st.session_state.df = df
    st.session_state.scaler = scaler
    st.session_state.feature_cols = feature_cols
    st.session_state.ready = True

# This block renders the main dashboard
# Reads processed data from session state
# Displays metrics and charts
# Allows individual customer analysis
# Core intelligence section
# Runs only when ready flag is set
if st.session_state.get("ready", False):

    df = st.session_state.df
    scaler = st.session_state.scaler
    feature_cols = st.session_state.feature_cols

    st.subheader("📊 Executive Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", len(df))
    c2.metric("Avg Spend", round(df["total_spend"].mean(), 2))
    c3.metric("Avg Orders", round(df["total_orders"].mean(), 2))
    c4.metric("Clusters", df["cluster"].nunique())

    st.subheader("📈 Overall Customer Distribution")

    cluster_counts = df["cluster"].value_counts().sort_index()
    labels = [CLUSTER_LABELS[c] for c in cluster_counts.index]
    values = cluster_counts.values
    total = values.sum()

    def autopct_func(pct):
        count = int(round(pct * total / 100.0))
        return f"{count} ({pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(values, labels=labels, autopct=autopct_func, startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    st.subheader("🔍 Individual Customer Intelligence")

    selected_customer = st.selectbox(
        "Select Customer",
        df["customer_id"].unique()
    )

    row = df[df["customer_id"] == selected_customer]
    X_row = row[feature_cols]
    X_scaled = scaler.transform(X_row)

    predicted_value = value_model.predict(X_scaled)[0]
    risk_prob = stability_model.predict_proba(X_scaled)[0][1]
    confidence = round(1 - risk_prob, 2)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cluster", CLUSTER_LABELS[int(row["cluster"].values[0])])
    m2.metric("Predicted Value", f"{predicted_value:,.2f}")
    m3.metric("Risk %", f"{risk_prob*100:.1f}")
    m4.metric("Confidence", confidence)

    st.progress(confidence)

    st.subheader("🎯 Recommended Action Strategy")
    st.success(
        f"""
        {CLUSTER_ACTIONS[int(row['cluster'].values[0])]}
        Risk Level: {'LOW' if risk_prob < 0.3 else 'MEDIUM' if risk_prob < 0.6 else 'HIGH'}
        Confidence Score: {confidence}
        """
    )

    st.subheader("⬇️ Download Intelligence Report")

    df["cluster_label"] = df["cluster"].map(CLUSTER_LABELS)

    st.download_button(
        "Download CSV Report",
        df.to_csv(index=False),
        file_name="customer_genome_intelligence_report.csv",
        mime="text/csv"
    )

st.markdown(
    "<h3 style='color:#22c55e; text-align:center;'>Developed by Anurag Kumar Singh</h3>",
    unsafe_allow_html=True
)
