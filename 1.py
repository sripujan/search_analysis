import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from trendspy import Trends

# Config
st.set_page_config(
    page_title="SEO TrendSpy Dashboard", page_icon="📈", layout="wide"
)


# Initialize TrendSpy client
@st.cache_resource
def get_trends_client():
  return Trends()


tr = get_trends_client()


def calculate_seo_score(series: pd.Series) -> dict:
  """Calculates an SEO Opportunity Score safely using native Python types."""
  if series is None or series.empty:
    return {"score": 0, "status": "No Data", "growth": 0.0, "avg": 0.0}

  # Clean data: drop missing values and ensure numeric
  clean_series = pd.to_numeric(series, errors="coerce").dropna()

  if clean_series.empty or len(clean_series) < 2 or clean_series.max() == 0:
    return {
        "score": 0,
        "status": "Insufficient Data",
        "growth": 0.0,
        "avg": 0.0,
    }

  # Extract numeric values as standard floats
  avg_vol = float(clean_series.mean())
  recent_vol = float(clean_series.tail(4).mean())
  past_vol = float(clean_series.head(4).mean())

  # Calculate growth rate percentage
  if past_vol > 0:
    growth = ((recent_vol - past_vol) / past_vol) * 100.0
  else:
    growth = recent_vol * 10.0

  std_dev = float(clean_series.std()) if len(clean_series) > 1 else 0.0

  # Calculate score
  raw_score = (avg_vol * 0.4) + (max(growth, 0.0) * 0.4) - (std_dev * 0.2)

  # Safe integer conversion
  if np.isnan(raw_score) or np.isinf(raw_score):
    normalized_score = 0
  else:
    normalized_score = int(min(max(raw_score, 0), 100))

  if normalized_score >= 70:
    status = "🔥 High Opportunity"
  elif normalized_score >= 40:
    status = "⚡ Moderate Potential"
  else:
    status = "❄️ Low Priority / Declining"

  return {
      "score": normalized_score,
      "status": status,
      "growth": round(growth, 2),
      "avg": round(avg_vol, 2),
  }

# Sidebar Layout
st.sidebar.title("🔍 TrendSpy Settings")
navigation = st.sidebar.radio(
    "Select Feature",
    ["Compare Keywords", "SEO Opportunity Score", "Real-Time Trends"],
)

# Region selection (strictly valid countries)
geo_code = st.sidebar.selectbox(
    "Region", ["US", "CN", "CA", "IN", "AU", "DE", "NP"], index=0
)

# -------------------------------------------------------------------
# Feature 1: Multi-Keyword Search Comparison
# -------------------------------------------------------------------
if navigation == "Compare Keywords":
  st.title("⚔️ Multi-Keyword Search Comparison")

  kw_input = st.text_input(
      "Enter keywords (comma-separated, max 5):",
      "Artificial Intelligence, Quantum Computing, Cybersecurity",
  )
  keywords = [k.strip() for k in kw_input.split(",") if k.strip()][:5]

  timeframe = st.selectbox(
      "Timeframe", ["today 12-m", "today 3-m", "now 7-d", "today 5-y"], index=0
  )

  if st.button("Analyze Interest"):
    with st.spinner("Fetching data from Google Trends..."):
      try:
        df = tr.interest_over_time(
            keywords=keywords, timeframe=timeframe, geo=geo_code
        )

        if df is not None and not df.empty:
          # Fill missing data
          df = df.fillna(0)

          st.subheader("Search Volume Comparison Over Time")
          fig = px.line(
              df,
              x=df.index,
              y=[c for c in df.columns if c in keywords],
              title=f"Relative Interest ({timeframe}) - {geo_code}",
              labels={"value": "Interest (0-100)", "variable": "Keyword"},
          )
          st.plotly_chart(fig, use_container_width=True)

          # Summary Metrics
          st.subheader("Summary Performance")
          cols = st.columns(len(keywords))
          for idx, kw in enumerate(keywords):
            if kw in df.columns:
              avg_val = round(float(df[kw].mean()), 1)
              max_val = int(df[kw].max())
              cols[idx].metric(kw, f"Avg: {avg_val}", f"Peak: {max_val}")
        else:
          st.error("No data returned for the specified terms.")
      except Exception as e:
        st.error(f"Error fetching comparison: {e}")

# -------------------------------------------------------------------
# Feature 2: SEO Opportunity Scoring
# -------------------------------------------------------------------
elif navigation == "SEO Opportunity Score":
  st.title("🎯 SEO Opportunity Scoring")

  kw = st.text_input("Enter Target Keyword:", "Generative AI").strip()

  if st.button("Run SEO Analysis"):
    with st.spinner(f"Analyzing '{kw}'..."):
      try:
        df = tr.interest_over_time(
            keywords=[kw], timeframe="today 12-m", geo=geo_code
        )

        if df is not None and not df.empty:
          df = df.fillna(0)

          # Get series dynamically
          matched_cols = [c for c in df.columns if c.lower() == kw.lower()]
          target_col = matched_cols[0] if matched_cols else df.columns[0]
          series = df[target_col]

          metrics = calculate_seo_score(series)

          st.markdown("---")
          c1, c2, c3 = st.columns(3)
          c1.metric("SEO Opportunity Score", f"{metrics['score']}/100")
          c2.metric("Rating", metrics["status"])
          c3.metric(
              "Annual Growth Rate",
              f"{metrics['growth']}%",
              delta=f"{metrics['growth']}%",
          )
          st.markdown("---")

          # Plot raw historical trend
          st.subheader(
              f"12-Month Search Interest Trend for '{kw}' ({geo_code})"
          )
          fig = px.line(
              df,
              x=df.index,
              y=target_col,
              labels={target_col: "Interest (0-100)", "index": "Date"},
          )
          st.plotly_chart(fig, use_container_width=True)

        else:
          st.error("Could not fetch valid trend data for this keyword.")
      except Exception as e:
        st.error(f"Analysis Error: {e}")

# -------------------------------------------------------------------
# Feature 3: Real-Time Trending Searches
# -------------------------------------------------------------------
elif navigation == "Real-Time Trends":
  st.title("⚡ Real-Time Trending Searches")
  st.write(
      "Discover what people are actively searching for right now in"
      f" *{geo_code}*."
  )

  if st.button("Fetch Current Trends"):
    with st.spinner("Downloading trending topics..."):
      try:
        trends_list = tr.trending_now(geo=geo_code)

        if trends_list is not None and len(trends_list) > 0:
          st.subheader(f"Top Trending Topics in {geo_code}")

          trends_df = pd.DataFrame(trends_list, columns=["Trending Keyword"])
          trends_df.index += 1

          st.dataframe(trends_df, use_container_width=True)
        else:
          st.warning(
              f"No real-time trends available right now for region '{geo_code}'."
          )

      except Exception as e:
        st.error(
            f"Error fetching real-time trends for '{geo_code}'. Google Trends"
            " may be throttling requests."
        )