import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client
import os
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image

load_dotenv()

st.set_page_config(page_title="OCC Fraud Dashboard", layout="wide", initial_sidebar_state="expanded")

USAA_COLORS = {
    "0.0": "#002F6C",
    "1.0": "#CC9900",
    "2.0": "#A7C7E7",
    "3.0": "#4D4D4F",
    "4.0": "#A7A8AA",
}

col1, col2 = st.columns(2)
# Supabase connection
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials missing. Add them to Modal secrets.")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load dataset
response = supabase.table("scrapeddata").select("*").execute()

if not response.data:
    st.warning("No data found in Supabase table 'websites'.")
    st.stop()

df = pd.DataFrame(response.data)

# Convert timestamps
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Dashboard Header
st.title("OCC Overview")
st.write(
    "Real-time analysis of fraud trends and clustering patterns with interactive search capabilities."
)

tab_dashboard, cluster_analysis, tab_search = st.tabs(["Dashboard", "Cluster Analysis", "Search Library"])

with tab_dashboard:
    
    col1,col2 = st.columns(2)
    with col1:

        if "date" in df.columns:
            df_counts = (
                df.groupby(df["date"].dt.date)
                .size()
                .reset_index(name="fraud_count")
            )

            fig_trend = px.line(
                df_counts,
                x="date",
                y="fraud_count",
                markers=True,
                title="Number of Fraud Reports Over Time"
            )
            st.plotly_chart(fig_trend)

    with col2:
        if "fraud_score" in df.columns and "date" in df.columns:
            df_score = (
                df.groupby(df["date"].dt.date)["fraud_score"]
                .mean()
                .reset_index(name="avg_fraud_score")
            )

            fig_score = px.line(
                df_score,
                x="date",
                y="avg_fraud_score",
                markers=True,
                title="Average Fraud Score Over Time"
            )
            st.plotly_chart(fig_score)

    col1, col2 = st.columns(2)
    with col1:

        if "date" in df.columns and "fraud_related" in df.columns:
            fraud_df = df[df["fraud_related"] == True].copy()
            fraud_df = fraud_df.set_index("date").sort_index()
            
            daily_counts = fraud_df.resample("D").size().rename("daily_fraud")

            rolling = daily_counts.rolling(30, min_periods=7).mean().reset_index()
            fig_roll = px.line(
                rolling,
                x="date",
                y="daily_fraud",
                title="Fraud Cases - 30 Day Rolling Average",
                labels={"daily_fraud": "Fraud Count"}
            )

            st.plotly_chart(fig_roll, use_container_width=True)
        else: 
            st.info("Insufficient data to calculate quarterly percent change.")

    with col2:
        fraud_df = df[df["fraud_related"] == True].copy()
        monthly = fraud_df.resample("M", on="date").size().reset_index(name = "fraud_count")

        monthly["pct_change"] = monthly["fraud_count"].pct_change() * 100

        fig_month_pct = px.bar(
            monthly,
            x="date",
            y="pct_change",
            title="Monthly Percent Change in Fraud Cases",
            labels={"pct_change": "Percent Change (%)"}
        )
        fig_month_pct.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_month_pct, use_container_width=True)

with cluster_analysis:
    col1, col2 = st.columns(2)
    with col1:
        if "kmeans_cluster" in df.columns and "fraud_score" in df.columns:

            heat_df = (
                df[df["fraud_related"] == True]
                .groupby("kmeans_cluster")["fraud_score"]
                .mean()
                .reset_index()
            )

            heat_df["cluster_str"] = heat_df["kmeans_cluster"].astype(str)

            fig_heat = px.bar(
                heat_df,
                x="cluster_str",
                y="fraud_score",
                title="Average Fraud Score by K-Means Cluster",
                color="cluster_str",
                color_discrete_map=USAA_COLORS,
                labels={"cluster_str": "K-Means Cluster", "fraud_score": "Average Fraud Score"},
            )

            st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        if {"kmeans_cluster", "date", "fraud_related"} <= set(df.columns):

            fraud_df = df[df["fraud_related"] == True].copy()

            if fraud_df.empty:
                st.info("No fraud data available for trend analysis.")
            else:
                cluster_trends = (
                    fraud_df.groupby([fraud_df["date"].dt.date, "kmeans_cluster"])
                    .size()
                    .reset_index(name="count")
                )

                cluster_trends["cluster_str"] = cluster_trends["kmeans_cluster"].astype(str)

                fig = px.area(
                    cluster_trends,
                    x="date",
                    y="count",
                    color="cluster_str",
                    color_discrete_map=USAA_COLORS,
                    labels={"count": "Cases", "cluster_str": "Cluster"},
                )

                st.plotly_chart(fig, use_container_width=True)

    # Word Clouds of Clusters 
    st.write("**Word Clouds**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("cluster_0.0_wordcloud.png", caption="Cluster 0 Word Cloud", use_container_width=True)
    with col2:
        st.image("cluster_1.0_wordcloud.png", caption="Cluster 1 Word Cloud", use_container_width=True)
    with col3: 
        st.image("cluster_2.0_wordcloud.png", caption="Cluster 2 Word Cloud", use_container_width=True)

with tab_search:

    st.header("Search Library")

    search_query = st.text_input("Enter a search term to explore your fraud dataset.")

    query = (search_query or "").strip()

    if query:
        mask = (
            (df["fraud_related"] == True) &
            (
                df["cleaned_text"].str.contains(query, case=False, na=False) |
                df["fraud_reason"].str.contains(query, case=False, na=False)
            )
        )

        results = df[mask]

        st.write(f"### {len(results)} results found")

        if results.empty:
            st.warning("No matches found.")
        else:
            for _, row in results.iterrows():
                with st.expander(row.get("type", "Fraud Entry")):
                    st.write("**ID:**", row.get("id"))
                    st.write("**Link:**", row.get("link"))
                    st.write("**Date:**", row.get("date"))
                    st.write("**Summary**")
                    st.write(row.get("summary"))

    else:
        st.text(" ")

# Footer
st.markdown("---")
st.markdown("*Dashboard last updated: {}*".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))