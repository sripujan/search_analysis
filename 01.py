import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from trendspy import Trends

# -------------------------------------------------------------------
# 1. Page Config & Session State
# -------------------------------------------------------------------
st.set_page_config(
    page_title="SEO TrendSpy Dashboard",
    page_icon="📈",
    layout="wide",
)

# Login Form
def login_form():
  _, col2, _ = st.columns([1, 2, 1])

  with col2:
    st.title("📈 TrendSpy Login")
    st.write("Enter your credentials to access the dashboard.")

    with st.form("login_form"):
      username = st.text_input("Username")
      password = st.text_input("Password", type="password")
      submit_button = st.form_submit_button("Sign In")

      if submit_button:
        if (
            username in USER_CREDENTIALS
            and USER_CREDENTIALS[username] == password
        ):
          st.session_state["logged_in"] = True
          st.session_state["user"] = username
          st.success("Login successful!")
          st.rerun()
        else:
          st.error("Invalid Username or Password")


def logout():
  st.session_state["logged_in"] = False
  st.session_state.pop("user", None)
  st.rerun()

# Config
st.set_page_config(
    page_title="SEO TrendSpy Dashboard", page_icon="📈", layout="wide"
)


# Initialize TrendSpy client
@st.cache_resource
def get_trends_client():
  return Trends()


tr = get_trends_client()


# CACHED DATA FETCHERS (Prevents 429 Rate Limits)
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_interest_over_time_cached(keywords_tuple, timeframe, geo_code):
  keywords = list(keywords_tuple)
  max_retries = 3
  backoff = 2

  for attempt in range(max_retries):
    try:
      df = tr.interest_over_time(
          keywords=keywords, timeframe=timeframe, geo=geo_code
      )
      return df
    except Exception as e:
      if "429" in str(e) and attempt < max_retries - 1:
        time.sleep(backoff)
        backoff *= 2
      else:
        raise e


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_trending_now_cached(geo_code):
  return tr.trending_now(geo=geo_code)


# -------------------------------------------------------------------
# ALGORITHM 1: K-RANK SCORING ALGORITHM
# -------------------------------------------------------------------
def calculate_k_rank(series: pd.Series) -> dict:
  """Calculates a custom K-Rank Opportunity Score (0-100) combining volume, momentum, and volatility stability."""
  if series is None or series.empty:
    return {"k_score": 0, "status": "No Data", "growth": 0.0, "avg_vol": 0.0}

  clean_series = pd.to_numeric(series, errors="coerce").dropna()

  if clean_series.empty or len(clean_series) < 2 or clean_series.max() == 0:
    return {
        "k_score": 0,
        "status": "Insufficient Data",
        "growth": 0.0,
        "avg_vol": 0.0,
    }

  # 1. Volume Factor
  avg_vol = float(clean_series.mean())

  # 2. Growth Factor (Short-term momentum)
  recent_vol = float(clean_series.tail(4).mean())
  past_vol = float(clean_series.head(4).mean())

  growth = (
      ((recent_vol - past_vol) / past_vol) * 100.0
      if past_vol > 0
      else recent_vol * 10.0
  )

  # 3. Volatility Factor (Lower std dev = higher rank reliability)
  std_dev = float(clean_series.std()) if len(clean_series) > 1 else 0.0

  # Composite K-Rank Weighted Formula
  volume_weight = 0.45
  growth_weight = 0.35
  stability_weight = 0.20

  stability_score = max(0, 100 - (std_dev * 2))
  raw_k_score = (
      (avg_vol * volume_weight)
      + (max(growth, 0.0) * growth_weight)
      + (stability_score * stability_weight)
  )

  # Safe integer clip
  k_score = int(min(max(raw_k_score, 0), 100))

  if k_score >= 70:
    status = "🚀 K-Rank Elite (High Priority)"
  elif k_score >= 45:
    status = "⚡ K-Rank Target (Good Opportunity)"
  else:
    status = "❄️ Low Priority Keyword"

  return {
      "k_score": k_score,
      "status": status,
      "growth": round(growth, 2),
      "avg_vol": round(avg_vol, 2),
  }


# -------------------------------------------------------------------
# ALGORITHM 2: NLP KEYWORD CLUSTERING (TF-IDF + K-Means)
# -------------------------------------------------------------------
def cluster_keywords_nlp(keyword_list: list, num_clusters: int = 3) -> pd.DataFrame:
  """Groups a list of target keywords into semantic topics using TF-IDF and K-Means Clustering."""
  if not keyword_list or len(keyword_list) < 2:
    return pd.DataFrame({"Keyword": keyword_list, "Cluster": [0] * len(keyword_list)})

  # Restrict max clusters to keyword list length
  actual_clusters = min(num_clusters, len(keyword_list))

  # Vectorize text features using TF-IDF
  vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
  tfidf_matrix = vectorizer.fit_transform(keyword_list)

  # Run K-Means Clustering
  kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
  kmeans.fit(tfidf_matrix)

  # Map outputs
  df_clusters = pd.DataFrame(
      {"Keyword": keyword_list, "Topic Cluster": [f"Cluster {c+1}" for c in kmeans.labels_]}
  )

  return df_clusters


# Sidebar Layout
st.sidebar.title("🔍 TrendSpy Settings")
navigation = st.sidebar.radio(
    "Select Feature",
    [
        "Compare Keywords",
        "K-Rank Keyword Scoring",
        "NLP Keyword Clustering",
        "Real-Time Trends",
    ],
)

geo_code = st.sidebar.selectbox(
    "Region", ["US", "NP", "IN", "AU", "DE"], index=0
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
        df = fetch_interest_over_time_cached(
            tuple(keywords), timeframe, geo_code
        )

        if df is not None and not df.empty:
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
        if "429" in str(e):
          st.error("⏳ Rate limit hit (429 Error). Please wait 2-3 minutes.")
        else:
          st.error(f"Error fetching comparison: {e}")

# -------------------------------------------------------------------
# Feature 2: K-Rank Scoring
# -------------------------------------------------------------------
elif navigation == "K-Rank Keyword Scoring":
  st.title("🎯 K-Rank Keyword Scoring Engine")

  kw = st.text_input("Enter Target Keyword:", "Generative AI").strip()

  if st.button("Calculate K-Rank Score"):
    with st.spinner(f"Evaluating K-Rank metric for '{kw}'..."):
      try:
        df = fetch_interest_over_time_cached((kw,), "today 12-m", geo_code)

        if df is not None and not df.empty:
          df = df.fillna(0)

          matched_cols = [c for c in df.columns if c.lower() == kw.lower()]
          target_col = matched_cols[0] if matched_cols else df.columns[0]
          series = df[target_col]

          metrics = calculate_k_rank(series)

          st.markdown("---")
          c1, c2, c3 = st.columns(3)
          c1.metric("K-Rank Score", f"{metrics['k_score']}/100")
          c2.metric("Opportunity Level", metrics["status"])
          c3.metric(
              "Momentum Growth Rate",
              f"{metrics['growth']}%",
              delta=f"{metrics['growth']}%",
          )
          st.markdown("---")

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
        if "429" in str(e):
          st.error("⏳ Rate limit hit (429 Error). Please wait a few minutes.")
        else:
          st.error(f"Analysis Error: {e}")

# -------------------------------------------------------------------
# Feature 3: NLP Keyword Clustering
# -------------------------------------------------------------------
elif navigation == "NLP Keyword Clustering":
  st.title("🧠 NLP Semantic Keyword Clustering")
  st.write(
      "Paste a list of keywords below to cluster them into semantic topical"
      " groups using **TF-IDF + K-Means**."
  )

  default_keywords = (
      "ai tools, ai machine learning, python for data science, python tutorial,"
      " machine learning python, cyber security tips, cloud security, network"
      " security"
  )
  raw_kw_input = st.text_area("Enter Keywords (comma-separated):", default_keywords)

  cluster_count = st.slider("Select Target Clusters", 2, 6, 3)

  if st.button("Run Clustering Model"):
    kw_list = [k.strip() for k in raw_kw_input.split(",") if k.strip()]

    if len(kw_list) < 2:
      st.warning("Please enter at least 2 keywords for clustering.")
    else:
      with st.spinner("Processing TF-IDF Vectorization & Clustering..."):
        clustered_df = cluster_keywords_nlp(kw_list, num_clusters=cluster_count)

        st.subheader("Grouped Topic Clusters")
        st.dataframe(clustered_df, use_container_width=True)

        # Plot cluster visual breakdown
        cluster_counts = (
            clustered_df["Topic Cluster"].value_counts().reset_index()
        )
        cluster_counts.columns = ["Topic Cluster", "Keyword Count"]

        fig = px.bar(
            cluster_counts,
            x="Topic Cluster",
            y="Keyword Count",
            title="Cluster Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# Feature 4: Real-Time Trending Searches
# -------------------------------------------------------------------
elif navigation == "Real-Time Trends":
  st.title("⚡ Real-Time Trending Searches")
  st.write(
      "Discover what people are actively searching for right now in"
      f" **{geo_code}**."
  )

  if st.button("Fetch Current Trends"):
    with st.spinner("Downloading trending topics..."):
      try:
        trends_list = fetch_trending_now_cached(geo_code)

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
        if "429" in str(e):
          st.error("⏳ Rate limit hit (429 Error). Please wait a few minutes.")
        else:
          st.error(f"Error fetching real-time trends for '{geo_code}': {e}")