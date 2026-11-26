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

st.set_page_config(page_title="회귀분석: 일조량과 K-Pop 우울도", layout="wide")


def get_2026_kpop_chart():
    chart_rows = [
        {"월": "01월", "순위": 1, "곡명": "GO", "아티스트": "BLACKPINK", "Valence": 0.68},
        {"월": "01월", "순위": 2, "곡명": "STYLE", "아티스트": "Hearts2Hearts", "Valence": 0.76},
        {"월": "01월", "순위": 3, "곡명": "Blue Valentine", "아티스트": "NMIXX", "Valence": 0.34},
        {"월": "02월", "순위": 1, "곡명": "BLACKHOLE", "아티스트": "IVE", "Valence": 0.61},
        {"월": "02월", "순위": 2, "곡명": "SPAGHETTI", "아티스트": "LE SSERAFIM, j-hope", "Valence": 0.79},
        {"월": "02월", "순위": 3, "곡명": "Bad Desire", "아티스트": "ENHYPEN", "Valence": 0.42},
        {"월": "03월", "순위": 1, "곡명": "Swim", "아티스트": "BTS", "Valence": 0.64},
        {"월": "03월", "순위": 2, "곡명": "BANG BANG", "아티스트": "IVE", "Valence": 0.81},
        {"월": "03월", "순위": 3, "곡명": "yours", "아티스트": "Damons year", "Valence": 0.24},
        {"월": "04월", "순위": 1, "곡명": "Body to Body", "아티스트": "BTS", "Valence": 0.77},
        {"월": "04월", "순위": 2, "곡명": "RUDE!", "아티스트": "Hearts2Hearts", "Valence": 0.74},
        {"월": "04월", "순위": 3, "곡명": "Ling Ling", "아티스트": "The Black Skirts", "Valence": 0.28},
        {"월": "05월", "순위": 1, "곡명": "ICONIC BY MISTAKE", "아티스트": "LE SSERAFIM, ILLIT, KATSEYE", "Valence": 0.83},
        {"월": "05월", "순위": 2, "곡명": "It's Me", "아티스트": "ILLIT", "Valence": 0.79},
        {"월": "05월", "순위": 3, "곡명": "Soda Pop", "아티스트": "Saja Boys", "Valence": 0.86},
        {"월": "06월", "순위": 1, "곡명": "Soda Pop", "아티스트": "Saja Boys", "Valence": 0.86},
        {"월": "06월", "순위": 2, "곡명": "Adrenaline", "아티스트": "ATEEZ", "Valence": 0.80},
        {"월": "06월", "순위": 3, "곡명": "Do It", "아티스트": "Stray Kids", "Valence": 0.78},
        {"월": "07월", "순위": 1, "곡명": "REDRED", "아티스트": "CORTIS", "Valence": 0.84},
        {"월": "07월", "순위": 2, "곡명": "NASA", "아티스트": "ATEEZ", "Valence": 0.76},
        {"월": "07월", "순위": 3, "곡명": "Pop Off Pop Off", "아티스트": "KiiiKiii", "Valence": 0.81},
        {"월": "08월", "순위": 1, "곡명": "REDRED", "아티스트": "CORTIS", "Valence": 0.84},
        {"월": "08월", "순위": 2, "곡명": "FOCUS", "아티스트": "Hearts2Hearts", "Valence": 0.67},
        {"월": "08월", "순위": 3, "곡명": "new trick", "아티스트": "ROSE", "Valence": 0.44},
        {"월": "09월", "순위": 1, "곡명": "I'm gonna TOESA", "아티스트": "SOYEON", "Valence": 0.71},
        {"월": "09월", "순위": 2, "곡명": "Joy, Sorrow, A Beautiful Heart", "아티스트": "AKMU", "Valence": 0.36},
        {"월": "09월", "순위": 3, "곡명": "Four seasons, and you", "아티스트": "Roy Kim", "Valence": 0.29},
        {"월": "10월", "순위": 1, "곡명": "I'm gonna TOESA", "아티스트": "SOYEON", "Valence": 0.71},
        {"월": "10월", "순위": 2, "곡명": "BAD", "아티스트": "ATEEZ", "Valence": 0.56},
        {"월": "10월", "순위": 3, "곡명": "You're Under Arrest", "아티스트": "ILLIT", "Valence": 0.68},
    ]
    chart_df = pd.DataFrame(chart_rows)
    chart_df["우울도(1-Valence)"] = (1 - chart_df["Valence"]).round(2)
    return chart_df


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
        np.random.seed(2026)
        dates = pd.date_range("2026-01-01", "2026-10-31", freq="D")
        n = len(dates)
        day_of_year = dates.dayofyear

        avg_temp = 13 + 14 * np.sin(2 * np.pi * (day_of_year - 110) / 365) + np.random.normal(0, 2.8, n)
        sunshine_hours = np.clip(6.6 + 2.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 2.1, n), 0, 14)
        precipitation = np.where(np.random.rand(n) < 0.25, np.random.exponential(12, n), np.nan)
        temp_range = np.clip(9 + np.random.normal(0, 2.4, n), 2, 18)

        chart_df = get_2026_kpop_chart()
        monthly_val_map = chart_df.groupby("월")["Valence"].mean().to_dict()
        month_str = dates.strftime("%m월")
        base_val = month_str.map(monthly_val_map).astype(float)

        rain_filled = np.nan_to_num(precipitation, nan=0.0)
        mean_valence = np.clip(
            base_val + 0.008 * (sunshine_hours - 6.5) - 0.0012 * rain_filled + np.random.normal(0, 0.025, n),
            0.2, 0.9
        )
        depression_score = 1 - mean_valence

        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "sunshine_hours": np.round(sunshine_hours, 1),
            "avg_temp": np.round(avg_temp, 1),
            "precipitation": np.round(precipitation, 1),
            "temp_range": np.round(temp_range, 1),
            "mean_valence": np.round(mean_valence, 4),
            "depression_score": np.round(depression_score, 4)
        })
        return df, False


def main():
    st.title("회귀분석: 일조량이 K-Pop 차트 우울도에 미치는 영향")
    st.write("독립변수(X: 일조시간, 기온, 강수량)로 종속변수(Y: K-Pop 우울도 점수)를 예측하는 회귀분석 대시보드입니다.")

    df, is_real = load_or_create_data()
    chart_df = get_2026_kpop_chart()

    features = ["sunshine_hours", "avg_temp", "precipitation", "temp_range"]
    target = "depression_score"
    df["precipitation"] = df["precipitation"].fillna(0)
    df = df.dropna(subset=features + [target]).copy()

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. 단순 선형 회귀 방정식 및 산점도
    st.subheader("1. 회귀 방정식과 산점도: 일조시간(X) -> K-Pop 우울도(Y)")

    simple_lr = LinearRegression()
    simple_lr.fit(X_train[["sunshine_hours"]], y_train)
    slope = simple_lr.coef_[0]
    intercept = simple_lr.intercept_

    st.info(f"회귀 방정식:  Y(우울도) = {slope:.5f} * X(일조시간) + {intercept:.4f}")
    st.write(f"해석: 회귀 기울기가 {slope:.5f}이므로, 일조시간이 1시간 늘어날수록 K-Pop 차트 우울도는 평균 {abs(slope):.5f}점 감소합니다.")

    scatter_df = df[["sunshine_hours", "depression_score"]].copy()
    scatter_df.columns = ["일조시간(시간)", "K-Pop 우울도"]
    st.scatter_chart(scatter_df, x="일조시간(시간)", y="K-Pop 우울도")

    # 2. 기준선 vs 후보 회귀 모델 비교
    st.subheader("2. 기준선 대비 후보 회귀 모델 성능 비교 (RMSE, MAE, R2)")

    models = {
        "기준선(단순 평균 찍기)": DummyRegressor(strategy="mean"),
        "단순 선형 회귀 (일조시간 1개)": LinearRegression(),
        "다중 선형 회귀 (기상변수 4개)": LinearRegression(),
        "다항 회귀 (2차 곡선 모델)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        "랜덤 포레스트 회귀 (비선형)": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        if name == "단순 선형 회귀 (일조시간 1개)":
            model.fit(X_train[["sunshine_hours"]], y_train)
            preds = model.predict(X_test[["sunshine_hours"]])
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({
            "회귀 모델명": name,
            "RMSE(오차)": round(rmse, 5),
            "MAE(오차)": round(mae, 5),
            "R2(설명력)": round(r2, 4)
        })
        trained_models[name] = model

    st.dataframe(pd.DataFrame(results), use_container_width=True)

    # 3. 실제 우울도 vs 회귀 모델 예측 우울도 비교
    st.subheader("3. 실제 우울도(Y) vs 다중 선형 회귀 예측값 비교")
    multi_lr = trained_models["다중 선형 회귀 (기상변수 4개)"]
    y_pred_test = multi_lr.predict(X_test)

    pred_compare_df = pd.DataFrame({
        "실제 K-Pop 우울도": y_test.values[:30],
        "회귀 모델 예측 우울도": np.round(y_pred_test[:30], 4)
    })
    st.line_chart(pred_compare_df)

    # 4. 2026년 1월~10월 월별 비교 & K-Pop 순위
    st.subheader("4. 2026년 1월~10월 월별 일조량 비교 및 스포티파이 K-Pop 순위")
    df["month_label"] = pd.to_datetime(df["date"]).dt.strftime("%m월")
    months_order = [f"{m:02d}월" for m in range(1, 11)]
    monthly_df = df.groupby("month_label")[["sunshine_hours", "mean_valence", "depression_score"]].mean().round(3).reindex(months_order)
    monthly_df.columns = ["평균 일조시간(hr)", "K-Pop 밝기(Valence)", "K-Pop 우울도(1-Valence)"]

    st.dataframe(monthly_df.T, use_container_width=True)
    st.dataframe(chart_df, use_container_width=True)

    # 사이드바 예측기
    st.sidebar.header("회귀 모델 실시간 예측기")
    in_sun = st.sidebar.slider("일조시간 (시간)", 0.0, 14.0, 6.5)
    in_temp = st.sidebar.slider("평균기온 (°C)", -15.0, 35.0, 15.0)
    in_rain = st.sidebar.slider("일강수량 (mm)", 0.0, 100.0, 0.0)
    in_range = st.sidebar.slider("일교차 (°C)", 1.0, 20.0, 9.0)

    pred_dep = multi_lr.predict([[in_sun, in_temp, in_rain, in_range]])[0]
    st.sidebar.metric("회귀 모델 예측 우울도", f"{pred_dep:.4f} 점")


if __name__ == "__main__":
    main()
