import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import datetime

st.set_page_config(page_title="국가별 증시 비교", layout="wide")

st.title("주요 국가별 증시 비교")
st.markdown(
    """
    이 앱은 대표 ETF/지수를 사용해 주요 국가(한국, 미국, 일본, 중국, 독일, 영국)의 증시 성과를 기간별로 비교합니다.

    데이터 출처: Yahoo Finance (yfinance)
    """
)

# --- 티커맵 (해당 국가를 대표하는 ETF/지수 티커)
TICKERS = {
    "한국 (EWY)": "EWY",        # iShares MSCI South Korea ETF
    "미국 (SPY)": "SPY",        # SPDR S&P 500 ETF
    "일본 (EWJ)": "EWJ",        # iShares MSCI Japan ETF
    "중국 (FXI)": "FXI",        # iShares China Large-Cap ETF
    "독일 (EWG)": "EWG",        # iShares MSCI Germany ETF
    "영국 (EWU)": "EWU",        # iShares MSCI United Kingdom ETF
    # 추가 국가 (선택적)
    "프랑스 (EWQ)": "EWQ",      # iShares MSCI France ETF
    "캐나다 (EWC)": "EWC",      # iShares MSCI Canada ETF
    "호주 (EWA)": "EWA",        # iShares MSCI Australia ETF
}

# Sidebar: 유저 입력
st.sidebar.header("설정")
default_countries = list(TICKERS.keys())
countries = st.sidebar.multiselect("비교할 국가 선택", options=default_countries, default=default_countries[:6])

today = datetime.date.today()
default_start = today - datetime.timedelta(days=365 * 5)
start_date = st.sidebar.date_input("시작일", default_start)
end_date = st.sidebar.date_input("종료일", today)

freq = st.sidebar.selectbox("빈도 (데이터 간격)", ["1d", "1wk", "1mo"], index=0)

normalize = st.sidebar.checkbox("기간 시작을 100으로 정규화 (비교용)", value=True)
show_corr = st.sidebar.checkbox("상관관계 히트맵 표시", value=True)
# 추가 입력: 무위험이자율(%) 및 벤치마크 선택
rf_percent = st.sidebar.number_input("무위험 이자율(연, %)", value=0.0, step=0.01, format="%.2f")
benchmark = st.sidebar.selectbox("기준 벤치 선택 (옵션)", options=["(없음)"] + default_countries, index=0)

@st.cache_data(ttl=60 * 60 * 6)
def download_data(tickers, start, end, interval="1d"):
    # yfinance: returns DataFrame with columns as tickers
    data = yf.download(list(tickers), start=start, end=end, interval=interval, progress=False)
    if data.empty:
        return pd.DataFrame()
    # some downloads return a multiindex (Adj Close)
    if ("Adj Close" in data.columns.get_level_values(0)) if hasattr(data.columns, 'levels') else False:
        data = data["Adj Close"]
    elif isinstance(data.columns, pd.MultiIndex):
        # try to pick Adj Close
        try:
            data = data[("Adj Close")]
        except Exception:
            # fall back to Close
            data = data["Close"]
    else:
        # single-index columns
        if "Adj Close" in data.columns:
            data = data["Adj Close"]
    # Ensure columns are tickers
    data = data.loc[:, ~data.columns.duplicated()]
    return data


if not countries:
    st.warning("비교할 국가를 하나 이상 선택하세요.")
    st.stop()

# Map selection to tickers
selected_tickers = {name: TICKERS[name] for name in countries}

with st.spinner("데이터를 다운로드하는 중입니다..."):
    raw = download_data(list(selected_tickers.values()), start_date, end_date + datetime.timedelta(days=1), interval=freq)

if raw.empty:
    st.error("선택 기간에 사용할 수 있는 데이터가 없습니다. 기간이나 빈도를 조정해 주세요.")
    st.stop()

# Rename columns to friendly country names
col_map = {v: k for k, v in selected_tickers.items()}
raw = raw.rename(columns=col_map)

# Fill missing data forward/backward (simple handling)
prices = raw.sort_index().ffill().bfill()

st.subheader("원시 종가 (Adj Close)")
st.dataframe(prices.tail(10))

# Normalized prices for comparison
if normalize:
    norm = prices.divide(prices.iloc[0]).multiply(100)
    chart_df = norm
    yaxis_title = "정규화된 지수 (시작=100)"
else:
    chart_df = prices
    yaxis_title = "가격"

st.subheader("가격 추이")
fig = px.line(chart_df, x=chart_df.index, y=chart_df.columns, labels={"value": yaxis_title, "index": "날짜"})
fig.update_layout(legend_title_text="국가")
st.plotly_chart(fig, use_container_width=True)

# 누적 수익률
st.subheader("기간 수익률 (누적)")
cum_returns = prices.pct_change().add(1).cumprod().iloc[-1].sub(1).multiply(100).sort_values(ascending=False)

# --- CAGR (기하평균 연환산 수익률) 계산
# 기간(년) 계산: 실제 날짜 차이를 이용 (보다 정확함)
years = (prices.index[-1] - prices.index[0]).days / 365.25 if len(prices.index) > 1 else np.nan
if years and years > 0:
    cagr = (prices.iloc[-1].divide(prices.iloc[0])).pow(1 / years).subtract(1).multiply(100)
else:
    cagr = pd.Series(np.nan, index=prices.columns)

# 연환산 변동성(%)
returns = prices.pct_change().dropna()
periods_per_year = 252 if freq == "1d" else (52 if freq == "1wk" else 12)
ann_vol = returns.std().multiply(np.sqrt(periods_per_year)).multiply(100)

# 색상: 양수는 녹색, 음수는 빨간색
bar_colors = ["green" if v >= 0 else "red" for v in cum_returns.values]

# 데이터프레임으로 만들어 hover/tooltip에 CAGR 포함
df_bar = pd.DataFrame({
    "국가": cum_returns.index,
    "누적수익률": cum_returns.values,
    "CAGR(%)": [cagr.get(k, np.nan).round(2) for k in cum_returns.index],
})

bar = px.bar(
    df_bar,
    x="국가",
    y="누적수익률",
    labels={"국가": "국가", "누적수익률": "누적 수익률 (%)"},
    text=df_bar["누적수익률"].round(2).astype(str) + "%",
    custom_data=[df_bar["CAGR(%)"]],
)
bar.update_traces(marker_color=bar_colors, textposition='outside', hovertemplate="국가: %{x}<br>누적수익률: %{y:.2f}%<br>CAGR: %{customdata[0]:.2f}%")
st.plotly_chart(bar, use_container_width=True)

if show_corr:
    st.subheader("일간 수익률 상관관계")
    returns = prices.pct_change().dropna()
    corr = returns.corr()
    hm = px.imshow(corr, x=corr.columns, y=corr.index, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    hm.update_layout(width=700, height=500)
    st.plotly_chart(hm)

st.subheader("요약 테이블")
latest = prices.iloc[-1]
returns = prices.pct_change().dropna()
change_1d = returns.iloc[-1].multiply(100)
change_total = prices.iloc[-1].divide(prices.iloc[0]).subtract(1).multiply(100)

# 연환산 관련: freq에 따라 연간 기간 수 설정
periods_per_year = 252 if freq == "1d" else (52 if freq == "1wk" else 12)
ann_return = returns.mean().multiply(periods_per_year).multiply(100)  # 단순화된 연환산 수익률(%)
ann_vol = returns.std().multiply(np.sqrt(periods_per_year)).multiply(100)  # 연환산 변동성(%)
sharpe = (ann_return - rf_percent) / ann_vol.replace(0, np.nan)

summary = pd.DataFrame({
    "최종가격": latest.round(2),
    "1단기등락(%)": change_1d.round(2),
    "기간 누적(%)": change_total.round(2),
    "연환산 수익률(%)": ann_return.round(2),
    "연환산 변동성(%)": ann_vol.round(2),
    "샤프비율": sharpe.round(3),
})
st.table(summary)

# 다운로드 버튼
csv = prices.to_csv()
st.download_button("가격 CSV 다운로드", csv, file_name="market_prices.csv", mime="text/csv")

st.markdown("---")
st.caption("참고: 이 데이터는 교육/분석용이며 투자 권유가 아닙니다. ETF는 지수와 완전히 동일하지 않을 수 있습니다.")

# 기준 벤치와 상대 성과 (옵션)
if benchmark and benchmark != "(없음)":
    # benchmark may be in friendly name format; ensure it's selected
    if benchmark not in prices.columns:
        st.warning("선택한 벤치가 현재 선택된 국가 목록에 없습니다. 벤치로 사용하려면 먼저 해당 국가를 선택하세요.")
    else:
        st.subheader(f"{benchmark} 대비 상대 성과 (정규화)")
        bench_series = prices[benchmark]
        rel = prices.divide(bench_series, axis=0)
        if normalize:
            rel = rel.divide(rel.iloc[0]).multiply(100)
            ylab = "기준 대비 정규화 (기준=100)"
        else:
            ylab = "기준 대비 비율"
        fig_rel = px.line(rel, x=rel.index, y=rel.columns, labels={"value": ylab, "index": "날짜"})
        st.plotly_chart(fig_rel, use_container_width=True)
