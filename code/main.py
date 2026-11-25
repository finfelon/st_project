import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정 (윈도우: Malgun Gothic / 맥: AppleGothic)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 1. 기상청(ASOS 일자료) CSV 불러오기 (기상청 파일은 cp949 인코딩)
weather = pd.read_csv("data/weather.csv", encoding="cp949")

# 필요한 열만 선택 후 이름 바꾸기
# (기상청 원본 열 이름: '일시', '합계 일조시간(hr)', '평균기온(°C)')
weather = weather[["일시", "합계 일조시간(hr)", "평균기온(°C)"]].rename(columns={
    "일시": "date",
    "합계 일조시간(hr)": "sunshine",
    "평균기온(°C)": "temp"
})

# 날짜에서 '월(1~12)' 뽑아내기
weather["date"] = pd.to_datetime(weather["date"])
weather["month"] = weather["date"].dt.month

# 2. 캐글(Spotify Charts) CSV 불러오기
spotify = pd.read_csv("data/spotify.csv", encoding="utf-8")

# 한국(South Korea) 차트만 남기기 (region 열이 있을 경우)
if "region" in spotify.columns:
    spotify = spotify[spotify["region"] == "South Korea"]

spotify["date"] = pd.to_datetime(spotify["date"])
spotify["month"] = spotify["date"].dt.month

# 3. 달(1~12월)별로 평균 계산하기
monthly_weather = weather.groupby("month")[["sunshine", "temp"]].mean()
monthly_music = spotify.groupby("month")[["valence", "energy"]].mean()

# 두 표를 'month(월)' 기준으로 합치기
monthly_df = pd.concat([monthly_weather, monthly_music], axis=1).round(3)

print("📅 [1월~12월 평균 일조량 및 차트 음악 밝기(Valence)]")
print(monthly_df)

# 4. 한눈에 보는 그래프 그리기 (막대: 일조량 / 꺾은선: 노래 밝기)
fig, ax1 = plt.subplots(figsize=(10, 5))

# 왼쪽 축: 월별 평균 일조시간 (주황색 막대)
ax1.bar(monthly_df.index, monthly_df["sunshine"], color="orange", alpha=0.5, label="평균 일조시간(시간)")
ax1.set_xlabel("월 (1월 ~ 12월)")
ax1.set_ylabel("하루 평균 일조시간 (시간)", color="darkorange")
ax1.set_xticks(range(1, 13))

# 오른쪽 축: 월별 차트 평균 Valence (파란색 꺾은선)
ax2 = ax1.twinx()
ax2.plot(monthly_df.index, monthly_df["valence"], color="royalblue", marker="o", linewidth=2.5, label="노래 밝기(Valence)")
ax2.set_ylabel("차트 평균 노래 밝기 (0=우울, 1=밝음)", color="royalblue")

plt.title("월별 일조시간과 대중음악 차트 정서(Valence) 비교")
plt.tight_layout()
plt.savefig("monthly_result.png", dpi=200)
plt.show()
