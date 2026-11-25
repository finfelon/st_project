import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# 한글 폰트 설정 (Windows: Malgun Gothic, Mac: AppleGothic, Colab/Linux: NanumGothic)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def load_or_create_data():
    """
    1. 데이터 수집 단계
    data/weather.csv 와 data/spotify.csv 가 있으면 불러오고,
    없을 경우 실제 데이터와 동일한 구조의 샘플 데이터를 생성합니다.
    """
    weather_path = "data/weather.csv"
    spotify_path = "data/spotify.csv"

    if os.path.exists(weather_path) and os.path.exists(spotify_path):
        print("✅ 실제 기상청 & 스포티파이 CSV 파일을 불러옵니다...")
        # 기상청 파일은 보통 cp949 인코딩 사용
        weather_df = pd.read_csv(weather_path, encoding="cp949")
        spotify_df = pd.read_csv(spotify_path, encoding="utf-8")

        # [주의] 실제 파일의 열 이름에 맞게 아래 컬럼명을 수정하세요!
        weather_df = weather_df.rename(columns={
            "일시": "date",
            "합계 일조시간(hr)": "sunshine_hours",
            "평균기온(°C)": "avg_temp",
            "일강수량(mm)": "precipitation",
            "최고기온(°C)": "max_temp",
            "최저기온(°C)": "min_temp"
        })
        weather_df["temp_range"] = weather_df["max_temp"] - weather_df["min_temp"]

        # 스포티파이 차트에서 한국(South Korea)만 필터링 (국가 열이 있는 경우)
        if "region" in spotify_df.columns:
            spotify_df = spotify_df[spotify_df["region"] == "South Korea"]

        # 일별 차트 평균 정서 점수 계산
        daily_music = spotify_df.groupby("date", as_index=False).agg(
            mean_valence=("valence", "mean"),
            mean_energy=("energy", "mean")
        )

        # 날짜 기준 병합
        df = pd.merge(weather_df, daily_music, on="date", how="inner")
        return df

    else:
        print("⚠️ data/ 폴더에 실제 CSV 파일이 없어, 테스트용 시뮬레이션 데이터를 생성합니다.")
        print("   (실제 제출 시 data/weather.csv 와 data/spotify.csv 를 넣어주세요!)\n")
        np.random.seed(42)
        dates = pd.date_range("2021-01-01", "2023-12-31", freq="D")
        n = len(dates)
        day_of_year = dates.dayofyear

        # 계절성을 반영한 기상 데이터 생성
        avg_temp = 13 + 14 * np.sin(2 * np.pi * (day_of_year - 110) / 365) + np.random.normal(0, 3, n)
        sunshine_hours = np.clip(6.5 + 2.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 2.2, n), 0, 14)
        precipitation = np.where(np.random.rand(n) < 0.25, np.random.exponential(12, n), np.nan)
        temp_range = np.clip(9 + np.random.normal(0, 2.5, n), 2, 18)

        # 일조량·강수량 + 연말(12월) 보상 원리 효과가 반영된 Valence 생성
        is_december = (dates.month == 12).astype(float)
        rain_filled = np.nan_to_num(precipitation, nan=0.0)
        mean_valence = np.clip(
            0.50 + 0.004 * sunshine_hours - 0.0012 * rain_filled + 0.045 * is_december + np.random.normal(0, 0.035, n),
            0.2, 0.85
        )
        mean_energy = np.clip(0.62 + 0.003 * sunshine_hours - 0.001 * rain_filled + np.random.normal(0, 0.04, n), 0.3, 0.9)

        return pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "sunshine_hours": np.round(sunshine_hours, 1),
            "avg_temp": np.round(avg_temp, 1),
            "precipitation": np.round(precipitation, 1),
            "temp_range": np.round(temp_range, 1),
            "mean_valence": np.round(mean_valence, 4),
            "mean_energy": np.round(mean_energy, 4)
        })


def main():
    # 1. 데이터 수집
    df = load_or_create_data()

    # 2. 변수 선택
    features = ["sunshine_hours", "avg_temp", "precipitation", "temp_range"]
    target = "mean_valence"

    # 3. 전처리 (비 안 온 날 강수량 결측치 -> 0으로 대체, 나머지 결측치 제거)
    df["precipitation"] = df["precipitation"].fillna(0)
    df = df.dropna(subset=features + [target]).copy()

    print(f"📊 [데이터 요약] 총 행 수: {len(df)}행 | 사용 열: {len(features) + 1}열")
    print(f"   - 평균 일조시간: {df['sunshine_hours'].mean():.2f}시간")
    print(f"   - 차트 평균 Valence: {df[target].mean():.4f} (표준편차: {df[target].std():.4f})\n")

    X = df[features]
    y = df[target]

    # 4. 분할 (훈련 80% : 평가 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. 기준선 (평균 모델) & 6. 후보 모델 정의
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

        results.append({"모델명": name, "RMSE(오차)": round(rmse, 5), "MAE": round(mae, 5), "R2(설명력)": round(r2, 4)})
        trained_models[name] = model

    results_df = pd.DataFrame(results)
    print("🏆 [5~8단계: 기준선 및 후보 모델 평가 결과]")
    print(results_df.to_string(index=False))

    # 7. 최종 모델 선택 (RMSE가 가장 낮은 모델 자동 선택)
    best_row = results_df.loc[results_df["RMSE(오차)"].idxmin()]
    baseline_rmse = results_df.loc[0, "RMSE(오차)"]
    improvement = (baseline_rmse - best_row["RMSE(오차)"]) / baseline_rmse * 100

    print(f"\n✅ [최종 선택 모델]: {best_row['모델명']}")
    print(f"   - 기준선 대비 RMSE 오차 감소율: {improvement:.2f}% 개선\n")

    # 9. 해석: 월별 패턴 및 변수 중요도 출력
    df["month"] = pd.to_datetime(df["date"]).dt.month
    monthly_summary = df.groupby("month")[["sunshine_hours", "mean_valence", "mean_energy"]].mean().round(4)
    print("📅 [월별 평균 일조시간 vs 음악 정서(Valence) 추이]")
    print(monthly_summary)

    rf_model = trained_models["랜덤 포레스트 회귀"]
    importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n🔍 [변수 중요도 (랜덤 포레스트 기준)]")
    for feat, imp in importances.items():
        print(f"   - {feat}: {imp:.4f}")

    # 결과 시각화 이미지 저장 (보고서 첨부용)
    os.makedirs("output", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 그래프 1: 일조시간 vs Valence 산점도 및 회귀선
    sns.regplot(data=df, x="sunshine_hours", y="mean_valence", ax=axes[0],
                scatter_kws={"alpha": 0.3, "color": "royalblue"}, line_kws={"color": "crimson"})
    axes[0].set_title("일조시간과 차트 평균 정서(Valence)의 관계")
    axes[0].set_xlabel("일조시간 (시간)")
    axes[0].set_ylabel("차트 평균 Valence (0=우울, 1=밝음)")

    # 그래프 2: 월별 일조시간과 평균 Valence 비교
    ax2 = axes[1].twinx()
    axes[1].bar(monthly_summary.index, monthly_summary["sunshine_hours"], alpha=0.35, color="orange", label="평균 일조시간(hr)")
    ax2.plot(monthly_summary.index, monthly_summary["mean_valence"], color="navy", marker="o", linewidth=2, label="평균 Valence")
    axes[1].set_title("월별 일조시간과 차트 음악 정서(Valence) 변화")
    axes[1].set_xlabel("월 (Month)")
    axes[1].set_ylabel("평균 일조시간 (시간)", color="darkorange")
    ax2.set_ylabel("차트 평균 Valence", color="navy")
    axes[1].set_xticks(range(1, 13))

    plt.tight_layout()
    plt.savefig("output/result_chart.png", dpi=200)
    print("\n🖼️ 그래프가 'output/result_chart.png'에 저장되었습니다!")


if __name__ == "__main__":
    main()
