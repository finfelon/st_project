import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# Streamlit 웹 페이지 기본 설정
st.set_page_config(page_title="햇빛과 플레이리스트", page_icon="☀️", layout="wide")


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
        is_real = True
        return df, is_real

    else:
        np.random.seed(42)
        dates = pd.date_range("2021-01-01", "2023-12-31", freq="D")
        n = len(dates)
        day_of_year = dates.dayofyear

        avg_temp = 13 + 14 * np.sin(2 * np.pi * (day_of_year - 110) / 365) + np.random.normal(0, 3, n)
        sunshine_hours = np.clip(6.5 + 2.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 2.2, n), 0, 14)
        precipitation = np.where(np.random.rand(n) < 0.25, np.random.exponential(12, n), np.nan)
        temp_range = np.clip(9 + np.random.normal(0, 2.5, n), 2, 18)

        is_december = (dates.month == 12).astype(float)
        rain_filled = np.nan_to_num(precipitation, nan=0.0)
        mean_valence = np.clip(
            0.50 + 0.004 * sunshine_hours - 0.0012 * rain_filled + 0.045 * is_december + np.random.normal(0, 0.035, n),
            0.2, 0.85
        )
        mean_energy = np.clip(0.62 + 0.003 * sunshine_hours - 0.001 * rain_filled + np.random.normal(0, 0.04, n), 0.3, 0.9)

        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "sunshine_hours": np.round(sunshine_hours, 1),
            "avg_temp": np.round(avg_temp, 1),
            "precipitation": np.round(precipitation, 1),
            "temp_range": np.round(temp_range, 1),
            "mean_valence": np.round(mean_valence, 4),
            "mean_energy": np.round(mean_energy, 4)
        })
        is_real = False
        return df, is_real


def main():
    st.title("☀️ 일조량과 대중음악 정서(Valence)의 관계 분석")
    st.write("햇빛이 줄어들고 추워지면 사람들은 정말 슬픈 노래를 더 많이 들을까? 회귀 분석으로 검증합니다.")

    df, is_real = load_or_create_data()
    if is_real:
        st.success("✅ 실제 기상청 & 스포티파이 CSV 데이터를 불러왔습니다.")
    else:
        st.info("ℹ️ 현재 테스트용 시뮬레이션 데이터로 실행 중입니다. (data/ 폴더에 CSV를 넣으면 자동 반영됩니다)")

    # 전처리 및 변수 설정
    features = ["sunshine_hours", "avg_temp", "precipitation", "temp_range"]
    target = "mean_valence"
    df["precipitation"] = df["precipitation"].fillna(0)
    df = df.dropna(subset=features + [target]).copy()

    X = df[features]
    y = df[target]

    # 데이터 분할 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 비교
    models = {
        "기준선(평균 모델)": DummyRegressor(strategy="mean"),
        "단순 선형 회귀(일조량만)": LinearRegression(),
        "다중 선형 회귀(기상 전체)": LinearRegression(),
        "다항 회귀(2차 곡선)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        "랜덤 포레스트 회귀": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        if name == "단순 선형 회귀(일조량만)":
            model.fit(X_train[["sunshine_hours"]], y_train)
            preds = model.predict(X_test[["sunshine_hours"]])
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({
            "모델명": name,
            "RMSE(오차)": round(rmse, 5),
            "MAE": round(mae, 5),
            "R2(설명력)": round(r2, 4)
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results)

    # 1. 모델 성능 비교 표 출력
    st.subheader("🏆 1. 기준선 및 후보 회귀 모델 비교 결과")
    st.dataframe(results_df, use_container_width=True)

    # 2. 월별 일조량 및 Valence 비교
    st.subheader("📅 2. 월별 평균 일조시간 vs 노래 밝기(Valence)")
    df["month"] = pd.to_datetime(df["date"]).dt.month
    monthly_df = df.groupby("month")[["sunshine_hours", "mean_valence", "mean_energy"]].mean().round(3)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**월별 평균 일조시간 (시간)**")
        st.bar_chart(monthly_df["sunshine_hours"])
    with col2:
        st.write("**월별 차트 평균 노래 밝기 (Valence)**")
        st.line_chart(monthly_df["mean_valence"])

    st.dataframe(monthly_df, use_container_width=True)

    # 3. 사이드바 시뮬레이터
    st.sidebar.header("🌦️ 오늘의 날씨로 노래 밝기 예측")
    in_sun = st.sidebar.slider("일조시간 (시간)", 0.0, 14.0, 6.5)
    in_temp = st.sidebar.slider("평균기온 (°C)", -15.0, 35.0, 15.0)
    in_rain = st.sidebar.slider("일강수량 (mm)", 0.0, 100.0, 0.0)
    in_range = st.sidebar.slider("일교차 (°C)", 1.0, 20.0, 9.0)

    rf_model = trained_models["랜덤 포레스트 회귀"]
    pred_val = rf_model.predict([[in_sun, in_temp, in_rain, in_range]])[0]
    st.sidebar.metric("예측된 차트 평균 Valence", f"{pred_val:.3f} 점")


if __name__ == "__main__":
    main()
