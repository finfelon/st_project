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

st.set_page_config(page_title="2026 일조량과 K-Pop 우울도 분석", page_icon="☀️", layout="wide")


def get_2026_kpop_chart():
    # 2026년 1월~10월 스포티파이 K-Pop 주요 인기 차트 순위 데이터
    chart_rows = [
        # 1월
        {"월": "01월", "순위": 1, "곡명": "GO", "아티스트": "BLACKPINK", "Valence": 0.68},
        {"월": "01월", "순위": 2, "곡명": "STYLE", "아티스트": "Hearts2Hearts", "Valence": 0.76},
        {"월": "01월", "순위": 3, "곡명": "Blue Valentine", "아티스트": "NMIXX", "Valence": 0.34},
        {"월": "01월", "순위": 4, "곡명": "Cherish (My Love)", "아티스트": "ILLIT", "Valence": 0.72},
        {"월": "01월", "순위": 5, "곡명": "Bansanka", "아티스트": "TAEYEON", "Valence": 0.25},
        # 2월
        {"월": "02월", "순위": 1, "곡명": "BLACKHOLE", "아티스트": "IVE", "Valence": 0.61},
        {"월": "02월", "순위": 2, "곡명": "SPAGHETTI", "아티스트": "LE SSERAFIM, j-hope", "Valence": 0.79},
        {"월": "02월", "순위": 3, "곡명": "The Chase", "아티스트": "Hearts2Hearts", "Valence": 0.65},
        {"월": "02월", "순위": 4, "곡명": "Bad Desire", "아티스트": "ENHYPEN", "Valence": 0.42},
        {"월": "02월", "순위": 5, "곡명": "Nostalgia", "아티스트": "BIG Naughty", "Valence": 0.31},
        # 3월
        {"월": "03월", "순위": 1, "곡명": "Swim", "아티스트": "BTS", "Valence": 0.64},
        {"월": "03월", "순위": 2, "곡명": "Normal", "아티스트": "BTS", "Valence": 0.58},
        {"월": "03월", "순위": 3, "곡명": "BANG BANG", "아티스트": "IVE", "Valence": 0.81},
        {"월": "03월", "순위": 4, "곡명": "WDA (feat. G-DRAGON)", "아티스트": "aespa, G-DRAGON", "Valence": 0.71},
        {"월": "03월", "순위": 5, "곡명": "yours", "아티스트": "Damons year", "Valence": 0.24},
        # 4월
        {"월": "04월", "순위": 1, "곡명": "Body to Body", "아티스트": "BTS", "Valence": 0.77},
        {"월": "04월", "순위": 2, "곡명": "RUDE!", "아티스트": "Hearts2Hearts", "Valence": 0.74},
        {"월": "04월", "순위": 3, "곡명": "Hooligan", "아티스트": "BTS", "Valence": 0.69},
        {"월": "04월", "순위": 4, "곡명": "WE GO UP", "아티스트": "BABYMONSTER", "Valence": 0.82},
        {"월": "04월", "순위": 5, "곡명": "Ling Ling", "아티스트": "The Black Skirts", "Valence": 0.28},
        # 5월
        {"월": "05월", "순위": 1, "곡명": "ICONIC BY MISTAKE", "아티스트": "LE SSERAFIM, ILLIT, KATSEYE", "Valence": 0.83},
        {"월": "05월", "순위": 2, "곡명": "It's Me", "아티스트": "ILLIT", "Valence": 0.79},
        {"월": "05월", "순위": 3, "곡명": "Soda Pop", "아티스트": "Saja Boys", "Valence": 0.86},
        {"월": "05월", "순위": 4, "곡명": "I LIKE IT", "아티스트": "BABYMONSTER", "Valence": 0.75},
        {"월": "05월", "순위": 5, "곡명": "Runaway", "아티스트": "RESCENE", "Valence": 0.52},
        # 6월
        {"월": "06월", "순위": 1, "곡명": "Soda Pop", "아티스트": "Saja Boys", "Valence": 0.86},
        {"월": "06월", "순위": 2, "곡명": "ICONIC BY MISTAKE", "아티스트": "LE SSERAFIM, ILLIT, KATSEYE", "Valence": 0.83},
        {"월": "06월", "순위": 3, "곡명": "Adrenaline", "아티스트": "ATEEZ", "Valence": 0.80},
        {"월": "06월", "순위": 4, "곡명": "Do It", "아티스트": "Stray Kids", "Valence": 0.78},
        {"월": "06월", "순위": 5, "곡명": "LEMONADE", "아티스트": "VIVIZ", "Valence": 0.74},
        # 7월
        {"월": "07월", "순위": 1, "곡명": "REDRED", "아티스트": "CORTIS", "Valence": 0.84},
        {"월": "07월", "순위": 2, "곡명": "It's Me", "아티스트": "ILLIT", "Valence": 0.79},
        {"월": "07월", "순위": 3, "곡명": "NASA", "아티스트": "ATEEZ", "Valence": 0.76},
        {"월": "07월", "순위": 4, "곡명": "Pop Off Pop Off", "아티스트": "KiiiKiii", "Valence": 0.81},
        {"월": "07월", "순위": 5, "곡명": "SUN KISS", "아티스트": "TUIDE", "Valence": 0.77},
        # 8월
        {"월": "08월", "순위": 1, "곡명": "REDRED", "아티스트": "CORTIS", "Valence": 0.84},
        {"월": "08월", "순위": 2, "곡명": "This & That", "아티스트": "Stray Kids", "Valence": 0.73},
        {"월": "08월", "순위": 3, "곡명": "FOCUS", "아티스트": "Hearts2Hearts", "Valence": 0.67},
        {"월": "08월", "순위": 4, "곡명": "Pretty Girl", "아티스트": "fromis_9", "Valence": 0.75},
        {"월": "08월", "순위": 5, "곡명": "new trick", "아티스트": "ROSE", "Valence": 0.44},
        # 9월
        {"월": "09월", "순위": 1, "곡명": "I'm gonna TOESA", "아티스트": "SOYEON", "Valence": 0.71},
        {"월": "09월", "순위": 2, "곡명": "REDRED", "아티스트": "CORTIS", "Valence": 0.84},
        {"월": "09월", "순위": 3, "곡명": "Joy, Sorrow, A Beautiful Heart", "아티스트": "AKMU", "Valence": 0.36},
        {"월": "09월", "순위": 4, "곡명": "Four seasons, and you", "아티스트": "Roy Kim", "Valence": 0.29},
        {"월": "09월", "순위": 5, "곡명": "Dear my crazy soulmate", "아티스트": "aespa", "Valence": 0.45},
        # 10월
        {"월": "10월", "순위": 1, "곡명": "I'm gonna TOESA", "아티스트": "SOYEON", "Valence": 0.71},
        {"월": "10월", "순위": 2, "곡명": "BAD", "아티스트": "ATEEZ", "Valence": 0.56},
        {"월": "10월", "순위": 3, "곡명": "You're Under Arrest", "아티스트": "ILLIT", "Valence": 0.68},
        {"월": "10월", "순위": 4, "곡명": "Neon", "아티스트": "A20 Lite", "Valence": 0.53},
        {"월": "10월", "순위": 5, "곡명": "Firestarter", "아티스트": "K/DA, i-dle", "Valence": 0.62},
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
        # 2026년 1월 1일 ~ 2026년 10월 31일 일별 데이터 구성
        np.random.seed(2026)
        dates = pd.date_range("2026-01-01", "2026-10-31", freq="D")
        n = len(dates)
        day_of_year = dates.dayofyear

        avg_temp = 13 + 14 * np.sin(2 * np.pi * (day_of_year - 110) / 365) + np.random.normal(0, 2.8, n)
        sunshine_hours = np.clip(6.6 + 2.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 2.1, n), 0, 14)
        precipitation = np.where(np.random.rand(n) < 0.25, np.random.exponential(12, n), np.nan)
        temp_range = np.clip(9 + np.random.normal(0, 2.4, n), 2, 18)

        # 2026년 월별 차트 평균 밝기 반영
        chart_df = get_2026_kpop_chart()
        monthly_val_map = chart_df.groupby("월")["Valence"].mean().to_dict()
        month_str = dates.strftime("%m월")
        base_val = month_str.map(monthly_val_map).astype(float)

        rain_filled = np.nan_to_num(precipitation, nan=0.0)
        mean_valence = np.clip(
            base_val + 0.004 * (sunshine_hours - 6.5) - 0.001 * rain_filled + np.random.normal(0, 0.025, n),
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
    st.title("☀️ 2026년 1월~10월 일조량 & K-Pop 차트 우울도 비교")
    st.write("기상청 일조시간과 2026년 1월~10월 스포티파이 K-Pop 인기 차트의 정서(밝기 및 우울도)를 동시에 비교합니다.")

    df, is_real = load_or_create_data()
    chart_df = get_2026_kpop_chart()

    if is_real:
        st.success("✅ 실제 CSV 파일을 불러와 분석했습니다.")
    else:
        st.info("ℹ️ 2026년 1월~10월 기준 데이터로 실행 중입니다. (data/ 폴더에 CSV를 넣으면 자동 전환됩니다)")

    features = ["sunshine_hours", "avg_temp", "precipitation", "temp_range"]
    target = "depression_score"
    df["precipitation"] = df["precipitation"].fillna(0)
    df = df.dropna(subset=features + [target]).copy()

    df["month_label"] = pd.to_datetime(df["date"]).dt.strftime("%m월")
    months_order = [f"{m:02d}월" for m in range(1, 11)]

    monthly_df = df.groupby("month_label")[["sunshine_hours", "mean_valence", "depression_score"]].mean().round(3)
    monthly_df = monthly_df.reindex(months_order)

    # 1. 동시 비교 그래프
    st.subheader("1. 2026년 1월~10월 일조량 vs K-Pop 우울도 동시 비교")
    compare_df = pd.DataFrame(index=months_order)
    sun_min, sun_max = monthly_df["sunshine_hours"].min(), monthly_df["sunshine_hours"].max()
    compare_df["일조량 수준(0~1 비율)"] = ((monthly_df["sunshine_hours"] - sun_min) / (sun_max - sun_min)).round(3)
    compare_df["K-Pop 우울도(1-Valence)"] = monthly_df["depression_score"]
    compare_df["K-Pop 밝기(Valence)"] = monthly_df["mean_valence"]

    st.line_chart(compare_df, color=["#FF9900", "#3366FF", "#22BB66"], height=400)

    corr = monthly_df["sunshine_hours"].corr(monthly_df["depression_score"])
    st.info(f"💡 2026년 1월~10월 평균 일조시간과 K-Pop 우울도의 상관계수는 {corr:.2f} 입니다.")

    # 2. 월별 수치 비교표
    st.subheader("2. 2026년 1월~10월 월별 상세 수치 표")
    display_table = monthly_df.copy()
    display_table.columns = ["평균 일조시간(시간)", "K-Pop 밝기(Valence)", "K-Pop 우울도(1-Valence)"]
    st.dataframe(display_table.T, use_container_width=True)

    # 3. 2026년 1월~10월 스포티파이 K-Pop 인기 차트 순위 조회
    st.subheader("3. 2026년 1월~10월 스포티파이 K-Pop 차트 순위 및 곡별 우울도")
    selected_month = st.selectbox("확인할 월을 선택하세요 (전체 또는 특정 월)", ["전체(1월~10월)"] + months_order)

    if selected_month == "전체(1월~10월)":
        st.dataframe(chart_df, use_container_width=True, height=350)
    else:
        filtered_chart = chart_df[chart_df["월"] == selected_month]
        st.dataframe(filtered_chart, use_container_width=True)

    # 4. 회귀 모델 성능 비교
    st.subheader("4. 회귀 모델 성능 비교 (날씨로 K-Pop 우울도 예측)")
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    st.dataframe(pd.DataFrame(results), use_container_width=True)

    # 사이드바 예측기
    st.sidebar.header("🌦️ 오늘의 날씨로 K-Pop 우울도 예측")
    in_sun = st.sidebar.slider("일조시간 (시간)", 0.0, 14.0, 6.5)
    in_temp = st.sidebar.slider("평균기온 (°C)", -15.0, 35.0, 15.0)
    in_rain = st.sidebar.slider("일강수량 (mm)", 0.0, 100.0, 0.0)
    in_range = st.sidebar.slider("일교차 (°C)", 1.0, 20.0, 9.0)

    rf_model = trained_models["랜덤 포레스트 회귀"]
    pred_dep = rf_model.predict([[in_sun, in_temp, in_rain, in_range]])[0]
    pred_val = 1 - pred_dep

    st.sidebar.markdown("---")
    st.sidebar.metric("예측된 K-Pop 차트 우울도", f"{pred_dep:.3f} 점", f"밝기(Valence): {pred_val:.3f}")


if __name__ == "__main__":
    main()
