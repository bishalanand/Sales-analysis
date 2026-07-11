import streamlit as st
import pandas as pd
import joblib

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# Load Data
# ==========================================

df = pd.read_csv("data/superstore_clean.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

forecast_output = pd.read_csv("models/forecast_output.csv")
model_metrics = pd.read_csv("models/model_metrics.csv")
comparison_table = pd.read_csv("models/model_comparison.csv")
anomaly_table = pd.read_csv("models/anomaly_table.csv")
cluster_table = pd.read_csv("models/cluster_table.csv")

xgb_model = joblib.load("models/xgb_model.pkl")

# ==========================================
# Sidebar Navigation
# ==========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)

# ==========================================
# Sales Overview
# ==========================================

if page == "Sales Overview":

    st.title("📊 Sales Overview Dashboard")

    # ==========================================
# Sales Overview
# ==========================================

if page == "Sales Overview":

    st.title("📊 Sales Overview Dashboard")

    # -------------------------------
    # Sidebar Filters
    # -------------------------------

    st.sidebar.header("Filters")

    region = st.sidebar.multiselect(
        "Select Region",
        options=sorted(df["Region"].unique()),
        default=sorted(df["Region"].unique())
    )

    category = st.sidebar.multiselect(
        "Select Category",
        options=sorted(df["Category"].unique()),
        default=sorted(df["Category"].unique())
    )

    # Filter Data
    filtered_df = df[
        (df["Region"].isin(region)) &
        (df["Category"].isin(category))
    ]

    # -------------------------------
    # KPI Cards
    # -------------------------------

    total_sales = filtered_df["Sales"].sum()
    total_orders = len(filtered_df)
    avg_sales = filtered_df["Sales"].mean()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Sales", f"${total_sales:,.0f}")
    c2.metric("Total Orders", f"{total_orders:,}")
    c3.metric("Average Sale", f"${avg_sales:,.2f}")

    st.markdown("---")

    # -------------------------------
    # Total Sales by Year
    # -------------------------------

    yearly_sales = (
        filtered_df
        .groupby(filtered_df["Order Date"].dt.year)["Sales"]
        .sum()
    )

    st.subheader("Total Sales by Year")

    st.bar_chart(yearly_sales)

    # -------------------------------
    # Monthly Sales Trend
    # -------------------------------

    monthly_sales = (
        filtered_df
        .groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
        .sum()
    )

    st.subheader("Monthly Sales Trend")

    st.line_chart(monthly_sales)

    # -------------------------------
    # Sales by Region
    # -------------------------------

    st.subheader("Sales by Region")

    region_sales = (
        filtered_df
        .groupby("Region")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(region_sales)

    # -------------------------------
    # Sales by Category
    # -------------------------------

    st.subheader("Sales by Category")

    category_sales = (
        filtered_df
        .groupby("Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(category_sales)

    # -------------------------------
    # Show Data
    # -------------------------------

    st.subheader("Filtered Dataset")

    st.dataframe(filtered_df)

# ==========================================
# Forecast Explorer
# ==========================================

# ==========================================
# Forecast Explorer
# ==========================================

elif page == "Forecast Explorer":

    st.title("📈 Forecast Explorer")

    st.write("Forecast generated using the best performing model (XGBoost).")

    # ----------------------------------
    # Select Forecast Type
    # ----------------------------------

    forecast_type = st.selectbox(
        "Forecast By",
        ["Category", "Region"]
    )

    # ----------------------------------
    # Select Category / Region
    # ----------------------------------

    if forecast_type == "Category":

        selected = st.selectbox(
            "Select Category",
            [
                "Furniture",
                "Technology",
                "Office Supplies"
            ]
        )

    else:

        selected = st.selectbox(
            "Select Region",
            [
                "West",
                "East"
            ]
        )

    # ----------------------------------
    # Forecast Horizon
    # ----------------------------------

    horizon = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

    # ----------------------------------
    # Forecast Values
    # ----------------------------------

    forecast = forecast_output[["Month", selected]].iloc[:horizon]

    st.subheader(f"{selected} Forecast")

    st.dataframe(
        forecast,
        use_container_width=True
    )

    # ----------------------------------
    # Forecast Chart
    # ----------------------------------

    chart = (
        forecast
        .set_index("Month")
    )

    st.line_chart(chart)

    # ----------------------------------
    # Model Performance
    # ----------------------------------

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    col1.metric(
        "MAE",
        f"{model_metrics.loc[0,'MAE']:.2f}"
    )

    col2.metric(
        "RMSE",
        f"{model_metrics.loc[0,'RMSE']:.2f}"
    )

    # ----------------------------------
    # Complete Forecast Table
    # ----------------------------------

    st.subheader("Complete Forecast Table")

    st.dataframe(
        forecast_output,
        use_container_width=True
    )

    # ----------------------------------
    # Model Comparison
    # ----------------------------------

    st.subheader("Model Comparison")

    st.dataframe(
        comparison_table,
        use_container_width=True
    )
# ==========================================
# Anomaly Report
# ==========================================

elif page == "Anomaly Report":

    st.title("🚨 Anomaly Report")

    st.write("Weekly sales anomalies detected using Isolation Forest and Rolling Z-Score.")

    # ----------------------------------
    # Select Detection Method
    # ----------------------------------

    method = st.radio(
        "Select Anomaly Detection Method",
        [
            "Isolation Forest",
            "Rolling Z-Score"
        ]
    )

    # ----------------------------------
    # Display Chart
    # ----------------------------------

    if method == "Isolation Forest":

        st.image(
            "models/anomaly_isolation.png",
            caption="Isolation Forest Anomaly Detection",
            use_container_width=True
        )

    else:

        st.image(
            "models/anomaly_zscore.png",
            caption="Rolling Z-Score Anomaly Detection",
            use_container_width=True
        )

    # ----------------------------------
    # Display Anomaly Table
    # ----------------------------------

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomaly_table,
        use_container_width=True
    )

    # ----------------------------------
    # Summary Statistics
    # ----------------------------------

    st.subheader("Summary")

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Weeks",
        len(anomaly_table)
    )

    if "IF_Anomaly" in anomaly_table.columns:

        col2.metric(
            "Isolation Forest Anomalies",
            anomaly_table["IF_Anomaly"].sum()
        )

    elif "IsolationForest" in anomaly_table.columns:

        col2.metric(
            "Isolation Forest Anomalies",
            anomaly_table["IsolationForest"].sum()
        )

    # ----------------------------------
    # Download Report
    # ----------------------------------

    csv = anomaly_table.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Anomaly Report",
        data=csv,
        file_name="anomaly_report.csv",
        mime="text/csv"
    )

# ==========================================
# Product Demand Segments
# ==========================================

elif page == "Product Demand Segments":

    st.title("📦 Product Demand Segments")

    st.write(
        "Products are segmented into demand groups using K-Means Clustering."
    )

    # Cluster Plot
    st.image(
        "models/cluster_plot.png",
        caption="K-Means Product Demand Clusters",
        use_container_width=True
    )

    # Cluster Filter
    clusters = sorted(cluster_table["Cluster"].unique())

    selected_cluster = st.selectbox(
        "Select Cluster",
        clusters
    )

    filtered_cluster = cluster_table[
        cluster_table["Cluster"] == selected_cluster
    ]

    # Summary
    st.subheader("Cluster Summary")

    col1, col2 = st.columns(2)

    col1.metric(
        "Products in Cluster",
        len(filtered_cluster)
    )

    col2.metric(
        "Total Clusters",
        cluster_table["Cluster"].nunique()
    )

    # Table
    st.subheader("Products in Selected Cluster")

    st.dataframe(
        filtered_cluster,
        use_container_width=True
    )

    # Distribution
    st.subheader("Cluster Distribution")

    cluster_count = (
        cluster_table["Cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(cluster_count)

    # Download
    csv = cluster_table.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Cluster Report",
        data=csv,
        file_name="cluster_report.csv",
        mime="text/csv"
    )