import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Customer Genome Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_models():
    behavior_model = joblib.load("models/behavior_model.pkl")
    value_model = joblib.load("models/value_model.pkl")
    stability_model = joblib.load("models/stability_model.pkl")
    return behavior_model, value_model, stability_model

behavior_model, value_model, stability_model = load_models()

st.sidebar.title("Input Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload Clean Transaction CSV",
    type=["csv"]
)

run_button = st.sidebar.button("Build Customer Intelligence")

st.title("Customer Genome Intelligence Dashboard")
st.write(
    "Single-page interactive dashboard combining clustering, prediction, "
    "risk assessment, similarity analysis, and confidence-aware decisions."
)

if uploaded_file is not None and run_button:

    df = pd.read_csv(uploaded_file)

    required_cols = {
        "customer_id",
        "invoice_no",
        "invoice_date",
        "quantity",
        "price",
        "transaction_value"
    }

    if not required_cols.issubset(df.columns):
        st.error("Uploaded file does not match required schema.")
        st.stop()

    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    customer_features = (
        df.groupby("customer_id")
          .agg(
              total_orders=("invoice_no", "nunique"),
              total_quantity=("quantity", "sum"),
              total_spend=("transaction_value", "sum"),
              avg_order_value=("transaction_value", "mean")
          )
          .reset_index()
    )

    activity_days = (
        df.groupby("customer_id")["invoice_date"]
          .apply(lambda x: (x.max() - x.min()).days)
          .reset_index(name="active_days")
    )

    customer_features = customer_features.merge(
        activity_days, on="customer_id", how="left"
    )

    feature_cols = [
        "total_orders",
        "total_quantity",
        "total_spend",
        "avg_order_value",
        "active_days"
    ]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_features[feature_cols])

    customer_features["cluster"] = behavior_model.predict(scaled_features)

    st.subheader("Executive Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", customer_features.shape[0])
    c2.metric("Avg Orders", round(customer_features["total_orders"].mean(), 2))
    c3.metric("Avg Spend", round(customer_features["total_spend"].mean(), 2))
    c4.metric("Clusters", customer_features["cluster"].nunique())

    st.divider()
    st.subheader("Customer Explorer")

    selected_customer = st.selectbox(
        "Select Customer ID",
        customer_features["customer_id"].unique()
    )

    row = customer_features[
        customer_features["customer_id"] == selected_customer
    ]

    X_row = row[feature_cols]
    X_row_scaled = scaler.transform(X_row)

    predicted_value = value_model.predict(X_row_scaled)[0]
    stability_prob = stability_model.predict_proba(X_row_scaled)[0][1]
    confidence_score = round(1 - stability_prob, 2)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cluster ID", int(row["cluster"].values[0]))
    m2.metric("Predicted Value", f"{predicted_value:,.2f}")
    m3.metric("Stability Risk", f"{stability_prob * 100:.1f}%")
    m4.metric("Confidence Score", confidence_score)

    st.progress(confidence_score)

    st.subheader("Similar Customers")

    similarity_scores = cosine_similarity(X_row_scaled, scaled_features)[0]
    customer_features["similarity"] = similarity_scores

    similar_customers = (
        customer_features
        .sort_values("similarity", ascending=False)
        .iloc[1:6][["customer_id", "similarity", "total_spend"]]
    )

    st.dataframe(similar_customers, use_container_width=True)

    st.subheader("Decision Recommendation")

    if stability_prob < 0.3:
        st.success("Low risk customer. Focus on loyalty programs and premium offers.")
    elif stability_prob < 0.6:
        st.warning("Moderate risk customer. Engagement and targeted promotions advised.")
    else:
        st.error("High risk customer. Immediate retention actions recommended.")

    st.divider()
    st.subheader("Export")

    export_df = row.copy()
    export_df["predicted_value"] = predicted_value
    export_df["stability_risk"] = stability_prob
    export_df["confidence_score"] = confidence_score

    st.download_button(
        label="Download Customer Intelligence Report",
        data=export_df.to_csv(index=False),
        file_name="customer_intelligence_report.csv",
        mime="text/csv"
    )
