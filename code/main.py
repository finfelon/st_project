
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

st.set_page_config(page_title="회귀분석: 일조량과 K-Pop 우울도", page_icon="📈", layout="wide")


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
    st.title("📈 회귀분석: 일조량이 K-Pop 차트 우울도에 미치는 영향")
    st.write("독립변수(X: 일조시간·기온·강수량)로 종속변수(Y: K-Pop 우울도 점수)를 예측하는 **회귀분석(Regression)** 대시보드입니다.")

    df, is_real = load_or_create_data()
    chart_df = get_2026_kpop_chart()

    features = ["sunshine_hours", "avg_temp", "precipitation", "temp_range"]
    target = "depression_score"
    df["precipitation"] = df["precipitation"].fillna(0)
    df = df.dropna(subset=features + [target]).copy()

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # [핵심 1] 단순 선형 회귀선(추세선) & 회귀 방정식 시각화
    # ---------------------------------------------------------
    st.subheader("1. 회귀 직선(추세선)과 회귀 방정식: 일조시간(X) -> K-Pop 우울도(Y)")

    simple_lr = LinearRegression()
    simple_lr.fit(X_train[["sunshine_hours"]], y_train)
    slope = simple_lr.coef_[0]
    intercept = simple_lr.intercept_

    col_eq1, col_eq2 = st.columns([2, 1])
    with col_eq1:
        st.latex(rf"Y(\text{{우울도}}) = {slope:.5f} \times X(\text{{일조시간}}) + {intercept:.4f}")
        if slope < 0:
            st.success(f"해석: 회귀 직선의 기울기가 음수({slope:.5f})입니다. 즉, 일조시간이 1시간 늘어날수록 K-Pop 차트 우울도는 평균 {abs(slope):.5f}점 감소합니다.")
        else:
            st.info(f"해석: 회귀 직선의 기울기가 양수({slope:.5f})입니다. 일조시간과 함께 계절 이벤트 영향이 반영되었습니다.")
    with col_eq2:
        st.metric("회귀 기울기 (Coefficient)", f"{slope:.5f}")
        st.metric("Y절편 (Intercept)", f"{intercept:.4f}")

    # 회귀선 데이터 생성 후 산점도와 함께 시각화
    x_grid = np.linspace(df["sunshine_hours"].min(), df["sunshine_hours"].max(), 50).reshape(-1, 1)
    y_line = simple_lr.predict(x_grid)

    reg_plot_df = pd.DataFrame({
        "일조시간(hr)": np.concatenate([df["sunshine_hours"].values, x_grid.flatten()]),
        "실제 데이터(산점도)": np.concatenate([df["depression_score"].values, [np.nan] * len(x_grid)]),
        "회귀 예측선(추세선)": np.concatenate([[np.nan] * len(df), y_line])
    })

    st.vega_lite_chart(
        reg_plot_df,
        {
            "height": 380,
            "layer": [
                {
                    "mark": {"type": "point", "filled": True, "opacity": 0.45, "color": "#4A90E2"},
                    "encoding": {
                        "x": {"field": "일조시간(hr)", "type": "quantitative", "title": "독립변수 X: 하루 일조시간 (시간)"},
                        "y": {"field": "실제 데이터(산점도)", "type": "quantitative", "title": "종속변수 Y: K-Pop 우울도 점수", "scale": {"zero": False}}
                    }
                },
                {
                    "mark": {"type": "line", "color": "#E63946", "strokeWidth": 3.5},
                    "encoding": {
                        "x": {"field": "일조시간(hr)", "type": "quantitative"},
                        "y": {"field": "회귀 예측선(추세선)", "type": "quantitative"}
                    }
                }
            ]
        },
        use_container_width=True
    )

    # ---------------------------------------------------------
    # [핵심 2] 기준선 vs 4가지 회귀 모델 성능 비교 표
    # ---------------------------------------------------------
    st.subheader("2. 기준선 대비 후보 회귀 모델 성능 비교 (RMSE · MAE · R2)")

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
            "RMSE (평균제곱근오차, 낮을수록 좋음)": round(rmse, 5),
            "MAE (평균절대오차, 낮을수록 좋음)": round(mae, 5),
            "R2 (결정계수, 높을수록 좋음)": round(r2, 4)
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

    # ---------------------------------------------------------
    # [핵심 3] 실제값 vs 회귀 모델 예측값 비교 그래프
    # ---------------------------------------------------------
    st.subheader("3. 실제 우울도(Y) vs 다중 선형 회귀 예측값(Y_hat) 비교")
    multi_lr = trained_models["다중 선형 회귀 (기상변수 4개)"]
    y_pred_test = multi_lr.predict(X_test)

    pred_compare_df = pd.DataFrame({
        "실제 K-Pop 우울도": y_test.values[:30],
        "회귀 모델 예측 우울도": np.round(y_pred_test[:30], 4)
    })
    st.caption("테스트 데이터 30일치에 대해 실제 우울도 점수와 회귀 모델이 예측한 점수를 나란히 비교한 그래프입니다.")
    st.line_chart(pred_compare_df, color=["#2B59C3", "#FF6B6B"], height=300)

    # ---------------------------------------------------------
    # [핵심 4] 다중 회귀 변수별 영향력(가중치) 비교
    # ---------------------------------------------------------
    st.subheader("4. 어떤 날씨 변수가 우울도에 가장 큰 영향을 줄까? (회귀 계수 비교)")
    coef_df = pd.DataFrame({
        "기상 변수": ["일조시간(hr)", "평균기온(°C)", "일강수량(mm)", "일교차(°C)"],
        "회귀 계수(기울기)": np.round(multi_lr.coef_, 5)
    }).set_index("기상 변수")
    st.bar_chart(coef_df)

    # ---------------------------------------------------------
    # [핵심 5] 2026년 1월~10월 월별 비교 & K-Pop 순위
    # ---------------------------------------------------------
    with st.expander("📅 2026년 1월~10월 월별 일조량 비교 및 스포티파이 K-Pop 순위 보기"):
        df["month_label"] = pd.to_datetime(df["date"]).dt.strftime("%m월")
        months_order = [f"{m:02d}월" for m in range(1, 11)]
        monthly_df = df.groupby("month_label")[["sunshine_hours", "mean_valence", "depression_score"]].mean().round(3).reindex(months_order)
        monthly_df.columns = ["평균 일조시간(hr)", "K-Pop 밝기(Valence)", "K-Pop 우울도(1-Valence)"]
        st.dataframe(monthly_df.T, use_container_width=True)
        st.dataframe(chart_df, use_container_width=True)

    # 사이드바: 실시간 회귀 예측기
    st.sidebar.header("📈 회귀 모델 실시간 예측기")
    in_sun = st.sidebar.slider("일조시간 (시간)", 0.0, 14.0, 6.5)
    in_temp = st.sidebar.slider("평균기온 (°C)", -15.0, 35.0, 15.0)
    in_rain = st.sidebar.slider("일강수량 (mm)", 0.0, 100.0, 0.0)
    in_range = st.sidebar.slider("일교차 (°C)", 1.0, 20.0, 9.0)

    pred_dep = multi_lr.predict([[in_sun, in_temp, in_rain, in_range]])[0]
    st.sidebar.metric("회귀 모델 예측 우울도", f"{pred_dep:.4f} 점")


if __name__ == "__main__":
    main()
