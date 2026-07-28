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

