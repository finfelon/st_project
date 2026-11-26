import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

st.set_page_config(page_title="햇빛과 K-Pop 우울도 동시 비교", page_icon="☀️", layout="wide")


@st.cache_data
def load_or_create_data():
    weather_path = "data/weather.csv"
    spotify_path = "data/spotify.csv"

    if os.path.exists(weather_path) and os.path.exists(spotify_path):
        weather_df = pd.read_csv(weather_path, encoding="cp949")
        spotify_df = pd.read_csv(spotify_path, encoding="utf-8")

        weather_df = weather_df.rename(columns={
            "일시": "date",
            "합계 일조시간(hr)": "sunshine_hours",
            "평균기온(°C)": "avg_temp",
            "일강수량(mm)": "precipitation",
            "최고기온(°C)": "max_temp",
            "최저기온(°C)": "min_temp"
        })
        weather_df["temp_range"] = weather_df["max_temp"] - weather_df["min_temp"]

        if "region" in spotify_df.columns:
            spotify_df = spotify_df[spotify_df["region"] == "South Korea"]

        daily_music = spotify_df.groupby("date", as_index=False).agg(
            mean_valence=("valence", "mean"),
            mean_energy=("energy", "mean")
        )

        df = pd.merge(weather_df, daily_music, on="date", how="inner")
        df["depression_score"] = (1 - df["mean_valence"]).round(4)
        return df, True

    else:
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        n = len(dates)
        day_of_year = dates.dayofyear

        avg_temp = 13 + 14 * np.sin(2 * np.pi * (day_of_year - 110) / 365) + np.random.normal(0, 3, n)
        sunshine_hours = np.clip(6.5 + 2.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 2.2, n), 0, 14)
        precipitation = np.where(np.random.rand(n) < 0.25, np.random.exponential(12, n), np.nan)
        temp_range = np.clip(9 + np.random.normal(0, 2.5, n), 2, 18)

        month = dates.month
        is_autumn_ballad = np.isin(month, [10, 11]).astype(float)
        is_december = (month == 12).astype(float)
        rain_filled = np.nan_to_num(precipitation, nan=0.0)

        mean_valence = np.clip(
            0.52 + 0.004 * sunshine_hours - 0.0012 * rain_filled - 0.04 * is_autumn_ballad + 0.05 * is_december + np.random.normal(0, 0.03, n),
            0.2, 0.85
        )
        mean_energy = np.clip(0.64 + 0.003 * sunshine_hours - 0.001 * rain_filled + np.random.normal(0, 0.035, n), 0.3, 0.9)
        depression_score = 1 - mean_valence

        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "sunshine_hours": np.round(sunshine_hours, 1),
            "avg_temp": np.round(avg_temp, 1),
            "precipitation": np.round(precipitation, 1),
            "temp_range": np.round(temp_range, 1),
            "mean_valence": np.round(mean_valence, 4),
            "depression_score": np.round(depression_score, 4),
            "mean_energy": np.round(mean_energy, 4)
        })
        return df, False


def main():
    st.title("☀️ 1월~12월 일조량 & K-Pop 차트 우울도 한눈에 비교하기")
    st.write("일조량(햇빛)과 K-Pop 차트 노래의 우울도·밝기를 **한 그래프 위에서 동시에 겹쳐서** 비교합니다.")

    df, is_real = load_or_create_data()
    if is_real:
        st.success("✅ 실제 기상청 & 스포티파이 K-Pop 차트 CSV 데이터를 불러왔습니다.")
    else:
        st.info("ℹ️ 현재 1월~12월 시뮬레이션 데이터로 실행 중입니다. (`data/weather.csv`, `data/spotify.csv`를 넣으면 실제 데이터로 전환됩니다)")

    features = ["sunshine_hours", "avg_temp", "precipitation", "temp_range"]
    target = "depression_score"
    df["precipitation"] = df["precipitation"].fillna(0)
    df = df.dropna(subset=features + [target]).copy()

    df["month"] = pd.to_datetime(df["date"]).dt.month
    monthly_df = df.groupby("month")[["sunshine_hours", "mean_valence", "depression_score"]].mean().round(3)
    monthly_df = monthly_df.reindex(range(1, 13))
    month_labels = [f"{m:02d}월" for m in range(1, 13)]
    monthly_df.index = month_labels

    st.subheader("📊 1. 일조량 vs K-Pop 우울도 &
