import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

# --- 페이지 설정 (가장 처음에 위치해야 함) ---
st.set_page_config(
    page_title="글로벌 증시 대시보드",
    page_icon="📈",
    layout="wide"
)

# --- (1) 데이터 정의: 티커 ---
# 사용자가 선택할 수 있는 주요 증시 지수 목록 (티커, 설명)
TICKERS = {
    "^GSPC": "S&P 500 (미국)",
    "^IXIC": "NASDAQ (미국)",
    "^KS11": "KOSPI (한국)",
    "^N225": "Nikkei 225 (일본)",
    "000001.SS": "상해종합 (중국)",
    "^FTSE": "FTSE 100 (영국)",
    "^GDAXI": "DAX (독일)",
}
TICKER_LABELS = list(TICKERS.values())
TICKER_SYMBOLS = list(TICKERS.keys())

# --- (2) 사이드바: 사용자 입력 (위젯) ---
st.sidebar.header("📈 옵션 선택")

# 1. 지수 선택 (Multi-select)
selected_labels = st.sidebar.multiselect(
    "비교할 지수를 선택하세요:",
    options=TICKER_LABELS,
    default=[TICKER_LABELS[0], TICKER_LABELS[2]]  # 기본값: S&P 500, KOSPI
)

# 선택된 레이블을 다시 티커 심볼로 변환
selected_symbols = [symbol for symbol, label in TICKERS.items() if label in selected_labels]

# 2. 기간 선택 (Date Input)
today = datetime.date.today()
one_year_ago = today - datetime.timedelta(days=365)

start_date = st.sidebar.date_input(
    "시작일",
    value=one_year_ago,
    max_value=today - datetime.timedelta(days=1)
)
end_date = st.sidebar.date_input(
    "종료일",
    value=today,
    max_value=today
)

# 날짜 유효성 검사
if start_date >= end_date:
    st.sidebar.error("오류: 종료일은 시작일보다 이후여야 합니다.")
    st.stop() # 오류 시 앱 실행 중지

# --- (3) 데이터 로딩 및 처리 ---

# 캐싱: 동일한 요청 시 데이터를 다시 불러오지 않도록 설정 (속도 향상)
@st.cache_data
def load_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end)["Adj Close"]
        # 컬럼 이름이 티커(e.g. ^KS11) 대신 레이블(e.g. KOSPI (한국))로 보이도록 변경
        if len(tickers) == 1:
            # yf.download가 1개 티커 요청 시 Series를 반환하는 경우 대비
            data = data.to_frame()
            data.columns = [TICKERS.get(tickers[0], tickers[0])]
        else:
            data = data.rename(columns=TICKERS)
        
        # 데이터가 없는 컬럼(e.g. 휴장일) 제거
        data = data.dropna(axis=1, how='all')
        
        return data
    except Exception as e:
        st.error(f"데이터 로딩 중 오류 발생: {e}")
        return pd.DataFrame()

# 선택된 항목이 있을 경우에만 데이터 로드
if selected_symbols:
    raw_data = load_data(selected_symbols, start_date, end_date)

    if raw_data.empty:
        st.warning("선택된 기간에 데이터가 없거나 로딩에 실패했습니다.")
    else:
        # --- (4) 메인 화면: 시각화 ---
        st.title("📈 주요국 증시 비교 대시보드")
        st.write(f"기간: **{start_date}** 부터 **{end_date}** 까지")

        # 1. 정규화된 차트 (수익률 비교)
        st.subheader("수익률 비교 (정규화된 차트)")
        st.write("선택한 기간의 시작일을 100으로 맞추어 수익률 추이를 비교합니다.")
        
        # 정규화 (시작일 기준으로 100으로 맞추기)
        # (현재 값 / 첫날 값) * 100
        try:
            # 데이터가 있는 첫 번째 날짜를 기준으로 정규화
            first_valid_idx = raw_data.apply(lambda col: col.first_valid_index()).max()
            if first_valid_idx is None:
                raise ValueError("데이터에 유효한 시작점이 없습니다.")
            
            normalized_data = (raw_data.loc[first_valid_idx:] / raw_data.loc[first_valid_idx:].iloc[0]) * 100
            st.line_chart(normalized_data)
            
        except Exception as e:
            st.error(f"정규화 차트 생성 중 오류: {e}")
            st.write("선택된 지수 중 하나가 해당 기간의 시작일에 데이터가 없을 수 있습니다.")


        # 2. 원본 데이터 차트 (주가 지수)
        st.subheader("원본 주가 지수")
        st.write("각 지수의 실제 종가(Adj Close) 추이입니다.")
        st.line_chart(raw_data)

        # 3. 원본 데이터 테이블
        st.subheader("원본 데이터 (DataFrame)")
        st.dataframe(raw_data.sort_index(ascending=False), use_container_width=True)

else:
    # 아무것도 선택하지 않았을 때의 초기 화면
    st.title("📈 주요국 증시 비교 대시보드")
    st.info("사이드바에서 비교할 지수와 기간을 선택해주세요.")
