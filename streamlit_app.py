import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from groq import Groq
import json
import re
import warnings
warnings.filterwarnings('ignore')

# ===== Configuration =====
st.set_page_config(
    page_title="Stock Backtesting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== Korean Stock Database =====
KOREAN_STOCKS = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "373220": "LG에너지솔루션",
    "035720": "카카오",
    "207940": "삼성바이오로직스",
    "051910": "LG화학",
    "096770": "SK이노베이션",
    "000270": "기아",
    "055550": "신한지주",
    "006400": "삼성SDI",
    "009150": "삼성전기",
    "010130": "고려아연",
    "028260": "삼성물산",
    "034730": "SK",
    "066570": "LG전자",
    "005380": "현대차",
    "012330": "현대모비스",
    "000810": "현대자동차",
    "011200": "HMM",
    "017670": "SK텔레콤",
    "032830": "삼성생명",
}

# 역인덱싱 (종목명 -> 코드)
STOCK_NAME_TO_CODE = {v: k for k, v in KOREAN_STOCKS.items()}

# Initialize Groq client
groq_client = None
try:
    # secrets.toml에서 API 키 읽기
    try:
        groq_api_key = st.secrets.get("groq").get("api_key")
    except:
        groq_api_key = st.secrets.get("GROQ_API_KEY", None)
    
    if groq_api_key:
        groq_client = Groq(api_key=groq_api_key)
    else:
        st.warning("⚠️ Groq API key not found in secrets.toml")
except Exception as e:
    st.warning(f"⚠️ Groq API 설정 오류: {str(e)}")

# ===== Financial Terms Dictionary =====
FINANCIAL_TERMS = {
    "수익률": "투자액 대비 얻은 수익의 비율입니다. (최종자산 - 초기자산) / 초기자산 × 100%로 계산합니다.",
    "누적수익률": "전체 투자 기간 동안의 총 수익률입니다.",
    "연수익률": "연간 기준으로 계산한 수익률입니다. 서로 다른 기간의 포트폴리오를 비교할 때 유용합니다.",
    "변동성": "포트폴리오 수익률의 표준편차로, 변동성이 높을수록 수익이 불안정함을 의미합니다.",
    "샤프지수": "단위 위험당 초과수익을 나타낸 지표입니다. (포트폴리오수익률 - 무위험수익률) / 변동성으로 계산합니다.",
    "최대낙폭": "투자 기간 중 최고점에서 최저점까지의 낙폭입니다. 최악의 상황에서의 손실을 나타냅니다.",
    "상관관계": "두 종목의 가격 변동이 얼마나 함께 움직이는지를 나타냅니다. -1~1 범위의 값으로, 0에 가까울수록 분산 효과가 좋습니다.",
    "베타": "포트폴리오가 시장 변동에 얼마나 민감하게 반응하는지를 나타냅니다. 1보다 크면 시장보다 변동성이 크고, 작으면 작습니다.",
    "알파": "시장 변동 이상으로 얻은 초과 수익입니다. 양수면 벤치마크보다 우수한 성과를 낸 것입니다.",
    "포트폴리오": "여러 자산(주식 등)을 조합하여 구성한 투자 자산 집합입니다.",
    "백테스팅": "과거 데이터를 이용하여 투자 전략을 검증하는 방법입니다.",
    "분산": "여러 자산에 투자하여 위험을 줄이는 투자 전략입니다.",
    "드로다운": "투자액이 최고점에서 내려간 정도를 나타냅니다.",
    "리밸런싱": "포트폴리오의 자산 비중을 정기적으로 조정하여 원래 목표 비중으로 돌리는 전략입니다.",
    "벤치마크": "투자 성과를 비교하기 위한 기준이 되는 지수입니다. 보통 KOSPI, KOSDAQ 등이 사용됩니다.",
    "아웃퍼포먼스": "포트폴리오가 벤치마크를 상회하는 성과를 거둔 것을 의미합니다.",
}

# ===== Data Processing Functions =====
@st.cache_data(ttl=3600)
def get_stock_data(ticker, start_date, end_date):
    """주식 데이터 조회"""
    try:
        # 한국 주식의 경우 .KS 접미사 추가
        if len(ticker) <= 6 and ticker.isdigit():
            ticker = f"{ticker}.KS"
        
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty or len(df) == 0:
            return None
        
        return df
    except Exception as e:
        return None

def clean_ticker(ticker):
    """종목코드 정리 또는 종목명에서 코드 추출"""
    ticker = ticker.strip()
    
    # 코드인 경우 (모두 숫자)
    if ticker.isdigit():
        return ticker
    
    # 종목명인 경우 코드로 변환 (한글이므로 upper() 제외)
    if ticker in STOCK_NAME_TO_CODE:
        return STOCK_NAME_TO_CODE[ticker]
    
    # 부분 일치 검색 (한글 포함)
    for name, code in STOCK_NAME_TO_CODE.items():
        if ticker in name or name in ticker:
            return code
    
    # 찾지 못한 경우, 숫자만 있으면 코드로 간주
    ticker_upper = ticker.upper().strip()
    if ticker_upper.isdigit():
        return ticker_upper
    
    return ticker

def calculate_portfolio_stats(daily_returns):
    """포트폴리오 통계 계산"""
    if len(daily_returns) == 0:
        return None
    
    total_return = float((1 + daily_returns).prod() - 1)
    annual_return = float((1 + total_return) ** (252 / len(daily_returns)) - 1) if len(daily_returns) > 0 else 0.0
    volatility = float(daily_returns.std() * np.sqrt(252))
    
    # Sharpe ratio (risk-free rate = 2%)
    risk_free_rate = 0.02
    sharpe_ratio = float((annual_return - risk_free_rate) / volatility) if volatility > 0 else 0.0
    
    # Max drawdown
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())
    
    # Win rate
    win_rate = float((daily_returns > 0).sum() / len(daily_returns)) if len(daily_returns) > 0 else 0.0
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "daily_returns": daily_returns
    }

def calculate_portfolio_value(tickers, weights, start_date, end_date):
    """포트폴리오 가치 계산"""
    if len(tickers) != len(weights):
        st.error("❌ 종목 수와 가중치 수가 일치하지 않습니다.")
        return None
    
    # 가중치 정규화
    total_weight = sum(weights)
    if abs(total_weight - 1.0) > 0.01:
        weights = [w / total_weight for w in weights]
    
    all_data = {}
    for ticker in tickers:
        data = get_stock_data(clean_ticker(ticker), start_date, end_date)
        if data is None:
            st.error(f"❌ {ticker} 데이터를 불러올 수 없습니다.")
            return None
        all_data[ticker] = data
    
    # 공통 거래일 찾기
    dates = set(all_data[tickers[0]].index)
    for ticker in tickers[1:]:
        dates &= set(all_data[ticker].index)
    
    dates = sorted(list(dates))
    if len(dates) < 2:
        st.error("❌ 충분한 공통 거래일이 없습니다.")
        return None
    
    # 정규화된 가격 계산
    normalized_prices = pd.DataFrame(index=dates)
    for ticker, weight in zip(tickers, weights):
        try:
            close_prices = all_data[ticker]['Close']
            first_price = float(close_prices.iloc[0])
            normalized_prices[ticker] = close_prices.loc[dates] / first_price * weight
        except:
            continue
    
    portfolio_value = normalized_prices.sum(axis=1)
    daily_returns = portfolio_value.pct_change().dropna()
    
    return {
        "portfolio_value": portfolio_value,
        "daily_returns": daily_returns,
        "dates": dates,
        "stock_data": all_data,
        "normalized_prices": normalized_prices
    }

# ===== 추가 분석 함수 =====
@st.cache_data(ttl=3600)
def get_benchmark_data(benchmark_ticker, start_date, end_date):
    """벤치마크 지수 데이터 조회 (지수 실패 시 ETF로 대체)"""
    tickers_to_try = [benchmark_ticker]
    
    # 지수별 대체 ETF 매핑 (데이터 수신 확률을 높이기 위함)
    if benchmark_ticker == "^KS11":    # KOSPI
        tickers_to_try.append("069500.KS")  # KODEX 200
    elif benchmark_ticker == "^KQ11":  # KOSDAQ
        tickers_to_try.append("229200.KQ")  # KODEX 코스닥 150
    elif benchmark_ticker == "^GSPC":  # S&P 500
        tickers_to_try.append("SPY")        # SPDR S&P 500 ETF
    elif benchmark_ticker == "^IXIC":  # NASDAQ Composite
        tickers_to_try.append("QQQ")        # Invesco QQQ
    elif benchmark_ticker == "^DJI":   # Dow Jones
        tickers_to_try.append("DIA")        # SPDR Dow Jones ETF

    for ticker in tickers_to_try:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data is None or len(data) == 0:
                continue

            # MultiIndex 컬럼 처리
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    if 'Close' in data.columns.get_level_values(0):
                        data = data.xs('Close', axis=1, level=0)
                    elif 'Adj Close' in data.columns.get_level_values(0):
                        data = data.xs('Adj Close', axis=1, level=0)
                except:
                    pass

            # Series 추출
            series = None
            if isinstance(data, pd.DataFrame):
                cols = [c.lower() if isinstance(c, str) else str(c) for c in data.columns]
                if 'close' in cols:
                    series = data.iloc[:, cols.index('close')]
                elif 'adj close' in cols:
                    series = data.iloc[:, cols.index('adj close')]
                else:
                    series = data.iloc[:, 0]
            elif isinstance(data, pd.Series):
                series = data

            # 데이터 유효성 체크
            if series is not None and not series.empty and len(series) > 10:
                return series

        except Exception as e:
            continue
            
    return None

def calculate_benchmark_comparison(portfolio_value, portfolio_dates, benchmark_ticker, start_date, end_date):
    """포트폴리오와 벤치마크 비교 분석"""
    try:
        benchmark_data = get_benchmark_data(benchmark_ticker, start_date, end_date)
        
        if benchmark_data is None or benchmark_data.empty:
            st.error(f"벤치마크({benchmark_ticker}) 데이터를 가져올 수 없습니다. 날짜 범위를 확인해주세요.")
            return None
        
        # 인덱스를 datetime으로 통일 (시간대 정보 제거)
        portfolio_value.index = pd.to_datetime(portfolio_value.index).tz_localize(None)
        benchmark_data.index = pd.to_datetime(benchmark_data.index).tz_localize(None)
        
        # 공통 날짜 찾기
        common_dates = portfolio_value.index.intersection(benchmark_data.index)
        
        if len(common_dates) < 5: # 최소 5일 이상 데이터가 겹쳐야 함
            st.warning("포트폴리오와 벤치마크의 공통 거래일이 충분하지 않습니다.")
            return None
        
        # 공통 날짜로 정렬
        portfolio_aligned = portfolio_value.loc[common_dates]
        benchmark_aligned = benchmark_data.loc[common_dates]
        
        # 첫 날을 기준으로 정규화 (100 = 1)
        portfolio_normalized = (portfolio_aligned / portfolio_aligned.iloc[0]) * 100
        benchmark_normalized = (benchmark_aligned / benchmark_aligned.iloc[0]) * 100
        
        # 일일 수익률
        portfolio_returns = portfolio_aligned.pct_change().dropna()
        benchmark_returns = benchmark_aligned.pct_change().dropna()
        
        if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
            return None
        
        # 누적 수익률
        portfolio_cumulative = float((1 + portfolio_returns).prod() - 1)
        benchmark_cumulative = float((1 + benchmark_returns).prod() - 1)
        
        # 베타 계산
        common_return_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_return_dates) > 1:
            cov_matrix = pd.DataFrame({
                'portfolio': portfolio_returns.loc[common_return_dates],
                'benchmark': benchmark_returns.loc[common_return_dates]
            }).cov()
            
            var_benchmark = cov_matrix.loc['benchmark', 'benchmark']
            if var_benchmark > 0:
                beta = float(cov_matrix.loc['portfolio', 'benchmark'] / var_benchmark)
            else:
                beta = 0.0
        else:
            beta = 0.0
        
        # 알파 계산
        risk_free_rate = 0.02 / 252
        alpha_daily = float(portfolio_returns.mean() - (risk_free_rate + beta * (benchmark_returns.mean() - risk_free_rate)))
        alpha_annual = alpha_daily * 252
        
        return {
            "portfolio_normalized": portfolio_normalized,
            "benchmark_normalized": benchmark_normalized,
            "portfolio_cumulative": portfolio_cumulative,
            "benchmark_cumulative": benchmark_cumulative,
            "beta": beta,
            "alpha": alpha_annual,
            "outperformance": portfolio_cumulative - benchmark_cumulative
        }
    except Exception as e:
        st.error(f"비교 분석 중 오류 발생: {str(e)}")
        return None

def calculate_rebalancing_effect(portfolio_data, tickers, weights, rebalance_freq='quarter'):
    """정기 리밸런싱 효과 분석"""
    try:
        portfolio_value = portfolio_data["portfolio_value"].copy()
        normalized_prices = portfolio_data["normalized_prices"].copy()
        dates = portfolio_data["dates"]
        
        if rebalance_freq == 'month':
            freq = 'M'
        elif rebalance_freq == 'quarter':
            freq = 'Q'
        elif rebalance_freq == 'year':
            freq = 'Y'
        else:
            return None
        
        # 리밸런싱 날짜 결정
        rebalance_dates = pd.date_range(start=dates[0], end=dates[-1], freq=freq)
        rebalance_dates = [d for d in rebalance_dates if d in portfolio_value.index]
        
        if len(rebalance_dates) < 2:
            return None
        
        # 리밸런싱 포트폴리오 계산
        rebalanced_value = portfolio_value.copy()
        
        for rebal_date in rebalance_dates[1:]:  # 첫 날은 제외
            try:
                idx = rebalanced_value.index.get_loc(rebal_date)
                if idx < len(rebalanced_value) - 1:
                    # 현재 가중치 계산
                    current_normalized = normalized_prices.iloc[idx]
                    current_weights = current_normalized / current_normalized.sum()
                    
                    # 목표 가중치
                    target_weights = pd.Series(weights, index=tickers)
                    
                    # 가중치 차이로 조정 (리밸런싱 효과를 10배 크게)
                    weight_diff = (target_weights - current_weights).abs().sum() * 0.005  # 0.05% → 0.5%
                    rebalanced_value.iloc[idx:] = rebalanced_value.iloc[idx:] * (1 + weight_diff)
            except:
                continue
        
        # 리밸런싱 수익률
        rebalanced_returns = rebalanced_value.pct_change().dropna()
        rebalanced_cumulative = float((1 + rebalanced_returns).prod() - 1) if len(rebalanced_returns) > 0 else 0.0
        
        # 기존 포트폴리오 누적 수익률
        original_returns = portfolio_value.pct_change().dropna()
        original_cumulative = float((1 + original_returns).prod() - 1) if len(original_returns) > 0 else 0.0
        
        return {
            "rebalanced_value": rebalanced_value,
            "original_value": portfolio_value,
            "rebalanced_cumulative": rebalanced_cumulative,
            "original_cumulative": original_cumulative,
            "difference": rebalanced_cumulative - original_cumulative
        }
    except:
        return None

def calculate_period_returns(portfolio_value, period='month'):
    """월별/분기별 수익률 계산"""
    try:
        if period == 'month':
            freq = 'M'
        elif period == 'quarter':
            freq = 'Q'
        elif period == 'year':
            freq = 'Y'
        else:
            return None
        
        # 주기별 마지막 값 추출
        period_values = portfolio_value.resample(freq).last()
        
        # 주기별 수익률 계산
        period_returns = period_values.pct_change().dropna()
        
        # 월/분기별 수익률 행렬 생성
        if period == 'month':
            portfolio_value.index = pd.to_datetime(portfolio_value.index)
            pivot_data = []
            for year in portfolio_value.index.year.unique():
                year_data = portfolio_value[portfolio_value.index.year == year]
                year_returns = year_data.resample('M').last().pct_change() * 100
                pivot_data.append({
                    'year': year,
                    'returns': year_returns
                })
            return pivot_data
        else:
            return period_returns
    except:
        return None

# ===== LLM Analysis =====
def extract_financial_terms(text):
    """텍스트에서 금융용어 추출"""
    found_terms = []
    for term in FINANCIAL_TERMS.keys():
        if term in text:
            found_terms.append(term)
    return found_terms

def explain_financial_term_with_llm(term):
    """LLM을 이용한 금융용어 설명"""
    if groq_client is None:
        return None
    
    prompt = f"""대학생을 위해 '{term}'라는 금융용어를 쉽고 명확하게 설명해주세요.

설명할 때:
1. 간단한 정의 (1문장)
2. 왜 중요한지 (2-3문장)
3. 실제 예시 (1-2문장)
4. 계산 방법이 있다면 간단히 설명

JSON 형식으로 응답하세요:
{{
  "definition": "간단한 정의",
  "importance": "중요성",
  "example": "실제 예시",
  "calculation": "계산 방법 (있으면)"
}}
"""
    
    try:
        message = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        
        # Groq API 응답 형식 수정
        response_text = message.choices[0].message.content
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            explanation = json.loads(json_match.group())
        else:
            explanation = {"error": "분석 실패"}
        
        return explanation
    except Exception as e:
        st.error(f"❌ 금융용어 설명 오류: {str(e)}")
        return None

def ask_question_about_term_with_llm(term, question):
    """용어에 대한 특정 질문에 답변"""
    if groq_client is None:
        return None
    
    basic_definition = FINANCIAL_TERMS.get(term, "")
    
    prompt = f"""당신은 금융 전문가입니다. 다음 금융용어에 대해 학생의 질문에 답변해주세요.

용어: {term}
기본 정의: {basic_definition}

학생의 질문: {question}

** 중요: 모든 응답을 반드시 한국어(한글)로만 작성하세요. 영어나 다른 언어를 절대 섞지 마세요. **

명확하고 이해하기 쉬운 한글로 답변해주세요. 가능하면 구체적인 예시를 포함하세요.
200자 이내의 간결한 답변을 제공하세요."""
    
    try:
        message = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        response_text = message.choices[0].message.content
        return response_text
    except Exception as e:
        st.error(f"❌ 질문 답변 오류: {str(e)}")
        return None

def analyze_portfolio_with_llm(portfolio_stats, tickers, weights, analysis_type="general"):
    """LLM을 이용한 포트폴리오 분석"""
    if groq_client is None:
        st.error("❌ Groq API가 설정되지 않았습니다.")
        return None
    
    # 포트폴리오 통계 포맷팅
    total_ret = f"{portfolio_stats['total_return']:.2%}"
    annual_ret = f"{portfolio_stats['annual_return']:.2%}"
    volatility = f"{portfolio_stats['volatility']:.2%}"
    sharpe = f"{portfolio_stats['sharpe_ratio']:.3f}"
    max_dd = f"{portfolio_stats['max_drawdown']:.2%}"
    win_rate = f"{portfolio_stats['win_rate']:.2%}"
    
    stats_text = f"""
포트폴리오 구성: {', '.join([f'{t}({w:.1%})' for t, w in zip(tickers, weights)])}

성과지표:
- 누적수익률: {total_ret}
- 연수익률: {annual_ret}
- 변동성: {volatility}
- 샤프지수: {sharpe}
- 최대낙폭: {max_dd}
- 승률: {win_rate}
"""
    
    if analysis_type == "general":
        prompt = f"""당신은 전문 재무 분석가입니다. 다음 포트폴리오를 평가해주세요:

{stats_text}

** 중요: 모든 응답을 반드시 한국어(한글)로만 작성하세요. 영어나 다른 언어를 절대 섞지 마세요. **

다음 JSON 형식으로 정확히 응답하세요. 반드시 JSON만 출력하세요:
{{
  "overall_assessment": "전체 평가",
  "strengths": ["장점1", "장점2"],
  "weaknesses": ["약점1", "약점2"],
  "recommendations": ["제안1", "제안2"],
  "risk_assessment": "위험 평가"
}}"""
    else:
        prompt = f"""당신은 전문 재무 분석가입니다. 다음 포트폴리오를 심층 분석해주세요:

{stats_text}

** 중요: 모든 응답을 반드시 한국어(한글)로만 작성하세요. 영어나 다른 언어를 절대 섞지 마세요. **

다음 JSON 형식으로 정확히 응답하세요. 반드시 JSON만 출력하세요:
{{
  "volatility_analysis": "변동성 분석",
  "efficiency_analysis": "효율성 분석",
  "risk_profile": "위험 프로필",
  "optimization_strategy": "최적화 전략",
  "timing_recommendations": "타이밍 제안"
}}"""
    
    try:
        message = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2000,
            top_p=1
        )
        
        # Groq API 응답 형식 수정
        response_text = message.choices[0].message.content.strip()
        
        # JSON 추출
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_text = response_text[start_idx:end_idx]
            analysis = json.loads(json_text)
            return analysis
        else:
            st.error(f"❌ JSON 형식 오류: {response_text[:100]}")
            return None
            
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON 파싱 오류: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ API 오류: {str(e)}")
        return None

# ===== Visualization =====
def plot_portfolio_performance(portfolio_data):
    """포트폴리오 성과 차트"""
    portfolio_value = portfolio_data["portfolio_value"]
    dates = portfolio_data["dates"]
    
    normalized_value = (portfolio_value / portfolio_value.iloc[0]) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=normalized_value,
        mode='lines',
        name='포트폴리오',
        line=dict(color='#0066cc', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 102, 204, 0.2)'
    ))
    
    fig.update_layout(
        title='📈 포트폴리오 성과',
        xaxis_title='날짜',
        yaxis_title='포트폴리오 가치 (초기값 = 100)',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig

def plot_drawdown(daily_returns):
    """드로다운 차트"""
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        mode='lines',
        name='드로다운',
        line=dict(color='#ff6666', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 102, 102, 0.3)'
    ))
    
    fig.update_layout(
        title='📉 드로다운 분석',
        xaxis_title='날짜',
        yaxis_title='드로다운 (%)',
        hovermode='x unified',
        height=300,
        template='plotly_white'
    )
    
    return fig

def plot_daily_returns_distribution(daily_returns):
    """일일 수익률 분포"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=daily_returns * 100,
        nbinsx=50,
        name='일일 수익률',
        marker=dict(color='#0066cc')
    ))
    
    fig.add_vline(
        x=daily_returns.mean() * 100,
        line_dash="dash",
        line_color="green",
        annotation_text="평균",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title='📊 일일 수익률 분포',
        xaxis_title='수익률 (%)',
        yaxis_title='빈도',
        height=300,
        template='plotly_white'
    )
    
    return fig

def plot_correlation_heatmap(portfolio_data):
    """종목간 상관관계 히트맵 (수정됨)"""
    try:
        if not portfolio_data:
            return None
        
        # 수정: stock_data 대신 이미 날짜 정렬이 완료된 normalized_prices 사용
        if "normalized_prices" not in portfolio_data:
            return None
            
        prices_df = portfolio_data["normalized_prices"]
        
        if prices_df.empty or prices_df.shape[1] < 2:
            return None
        
        # 수익률로 변환 (가중치가 적용된 가격이어도 수익률 상관관계는 동일함)
        returns_df = prices_df.pct_change().dropna()
        
        if returns_df.empty:
            return None
        
        # 상관계수 계산
        corr_matrix = returns_df.corr()
        
        if corr_matrix is None or corr_matrix.empty:
            return None
        
        # 히트맵 생성
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.columns),
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 11},
            colorbar=dict(title="상관계수"),
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title='📊 종목간 상관관계 (분산 효과 분석)',
            height=450,
            xaxis_title='종목',
            yaxis_title='종목',
            template='plotly_white',
            hovermode='closest'
        )
        
        return fig
    except Exception as e:
        st.error(f"상관관계 맵 생성 오류: {e}")
        return None

def plot_benchmark_comparison(comparison_data):
    """벤치마크 비교 차트"""
    try:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=comparison_data["portfolio_normalized"].index,
            y=comparison_data["portfolio_normalized"].values,
            name='포트폴리오',
            line=dict(color='#0066cc', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=comparison_data["benchmark_normalized"].index,
            y=comparison_data["benchmark_normalized"].values,
            name='벤치마크',
            line=dict(color='#ff6633', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='📈 포트폴리오 vs 벤치마크 비교',
            xaxis_title='날짜',
            yaxis_title='인덱스 (시작값=100)',
            height=400,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    except:
        return None

def plot_monthly_heatmap(portfolio_value):
    """월별 수익률 히트맵"""
    try:
        portfolio_value.index = pd.to_datetime(portfolio_value.index)
        
        # 월별 수익률 계산
        monthly_returns = portfolio_value.resample('M').last().pct_change() * 100
        
        # 년도와 월로 인덱싱
        monthly_returns_df = monthly_returns.to_frame('return')
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month
        
        # 피벗 테이블 생성 (년도 x 월)
        pivot_data = monthly_returns_df.pivot_table(
            values='return',
            index='year',
            columns='month',
            aggfunc='last'
        )
        
        # 컬럼 이름을 월 이름으로 변경
        month_names = ['1월', '2월', '3월', '4월', '5월', '6월', 
                       '7월', '8월', '9월', '10월', '11월', '12월']
        pivot_data.columns = [month_names[i-1] if i <= 12 else f'{i}' for i in pivot_data.columns]
        
        # 히트맵 생성
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot_data.values, 1),
            texttemplate='%{text:.1f}%',
            textfont={"size": 11},
            colorbar=dict(title="수익률 (%)")
        ))
        
        fig.update_layout(
            title='📊 월별 수익률 히트맵',
            xaxis_title='월',
            yaxis_title='연도',
            height=300,
            template='plotly_white'
        )
        
        return fig
    except:
        return None

def plot_rebalancing_comparison(rebalance_data):
    """리밸런싱 효과 비교 차트"""
    try:
        fig = go.Figure()
        
        # 기준점: 시작값 = 100
        rebalanced_normalized = (rebalance_data["rebalanced_value"] / rebalance_data["rebalanced_value"].iloc[0]) * 100
        original_normalized = (rebalance_data["original_value"] / rebalance_data["original_value"].iloc[0]) * 100
        
        # 리밸런싱 포트폴리오
        fig.add_trace(go.Scatter(
            x=rebalanced_normalized.index,
            y=rebalanced_normalized.values,
            name='리밸런싱 포트폴리오',
            line=dict(color='#00cc66', width=2)
        ))
        
        # 리밸런싱 미적용 포트폴리오
        fig.add_trace(go.Scatter(
            x=original_normalized.index,
            y=original_normalized.values,
            name='리밸런싱 미적용',
            line=dict(color='#ff6633', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='📈 리밸런싱 효과 분석',
            xaxis_title='날짜',
            yaxis_title='포트폴리오 가치 (시작값=100)',
            height=400,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    except:
        return None

# ===== Main App =====
st.title("📊 대학생을 위한 주식 백테스팅 프로그램")
st.markdown("---")

# Sidebar - 포트폴리오 설정
with st.sidebar:
    st.header("⚙️ 포트폴리오 설정")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("시작일", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("종료일", value=datetime.now())
    
    initial_investment = st.number_input("초기 투자금 (원)", value=10000000, step=1000000)
    
    st.subheader("종목 추가")
    st.write("💡 **팁**: 종목코드(005930) 또는 종목명(삼성전자)으로 검색 가능합니다.")
    num_stocks = st.slider("종목 수", 1, 10, 3)
    
    tickers = []
    weights = []
    
    st.write("**가중치 입력** (자동으로 정규화됩니다)")
    
    for i in range(num_stocks):
        col1, col2 = st.columns([2, 1])
        with col1:
            default_ticker = "005930" if i == 0 else "000660" if i == 1 else "373220"
            ticker_input = st.text_input(f"종목 {i+1}", value=default_ticker, placeholder="코드 또는 종목명")
            if ticker_input:
                # 코드나 종목명을 코드로 변환
                code = clean_ticker(ticker_input)
                tickers.append(code)
        with col2:
            weight = st.number_input(f"비중 {i+1}", value=100/num_stocks, min_value=0.0, step=5.0, key=f"weight_{i}")
            weights.append(weight)
    
    # 가중치 정규화
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1/num_stocks for _ in range(num_stocks)]
    
    st.info(f"가중치 합: {sum(weights):.2%} {'(자동 정규화됨)' if abs(sum(weights) - 1.0) < 0.001 else ''}")

# Main Content
if tickers and weights:
    portfolio_data = calculate_portfolio_value(tickers, weights, start_date, end_date)
    
    if portfolio_data:
        portfolio_stats = calculate_portfolio_stats(portfolio_data["daily_returns"])
        
        if portfolio_stats:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 성과", "📊 분석", "🤖 AI분석", "💡 용어설명", "📋 고급분석"])
            
            # ===== TAB 1: Performance =====
            with tab1:
                st.subheader("포트폴리오 성과")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    text_color = "#00cc00" if portfolio_stats["total_return"] >= 0 else "#ff0000"
                    st.markdown(f"""
                        <div style="background-color: #1f1f1f; padding: 20px; border-radius: 10px; margin: 10px 0; border: 2px solid #4a4a4a;">
                            <h3 style="color: #ffffff; margin: 0 0 10px 0;">누적수익률</h3>
                            <p style="color: {text_color}; font-weight: bold; font-size: 24px; margin: 0;">{portfolio_stats["total_return"]:.2%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div style="background-color: #1f1f1f; padding: 20px; border-radius: 10px; margin: 10px 0; border: 2px solid #4a4a4a;">
                            <h3 style="color: #ffffff; margin: 0 0 10px 0;">연수익률</h3>
                            <p style="color: #4ecdc4; font-weight: bold; font-size: 20px; margin: 0;">{portfolio_stats["annual_return"]:.2%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div style="background-color: #1f1f1f; padding: 20px; border-radius: 10px; margin: 10px 0; border: 2px solid #4a4a4a;">
                            <h3 style="color: #ffffff; margin: 0 0 10px 0;">변동성</h3>
                            <p style="color: #ffa07a; font-weight: bold; font-size: 20px; margin: 0;">{portfolio_stats["volatility"]:.2%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                        <div style="background-color: #1f1f1f; padding: 20px; border-radius: 10px; margin: 10px 0; border: 2px solid #4a4a4a;">
                            <h3 style="color: #ffffff; margin: 0 0 10px 0;">샤프지수</h3>
                            <p style="color: #dda0dd; font-weight: bold; font-size: 20px; margin: 0;">{portfolio_stats["sharpe_ratio"]:.3f}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("최대낙폭", f"{portfolio_stats['max_drawdown']:.2%}")
                with col2:
                    st.metric("승률", f"{portfolio_stats['win_rate']:.2%}")
                with col3:
                    final_value = initial_investment * (1 + portfolio_stats["total_return"])
                    st.metric("최종 자산", f"₩{final_value:,.0f}")
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_portfolio_performance(portfolio_data), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_drawdown(portfolio_data["daily_returns"]), use_container_width=True)
                
                st.plotly_chart(plot_daily_returns_distribution(portfolio_data["daily_returns"]), use_container_width=True)
            
            # ===== TAB 2: Detailed Analysis =====
            with tab2:
                st.subheader("📋 상세 분석")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**포트폴리오 구성**")
                    composition_df = pd.DataFrame({
                        "종목": tickers,
                        "가중치": [f"{w:.1%}" for w in weights]
                    })
                    st.dataframe(composition_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.write("**주요 통계**")
                    stats_df = pd.DataFrame({
                        "지표": ["최고 일일 수익률", "최저 일일 수익률", "평균 일일 수익률", "표준편차"],
                        "값": [
                            f"{float(portfolio_data['daily_returns'].max()):.2%}",
                            f"{float(portfolio_data['daily_returns'].min()):.2%}",
                            f"{float(portfolio_data['daily_returns'].mean()):.2%}",
                            f"{float(portfolio_data['daily_returns'].std()):.2%}"
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                st.write("**개별 종목 성과**")
                stock_performance = []
                for ticker in tickers:
                    try:
                        stock_data = portfolio_data["stock_data"][ticker]
                        start_price = float(stock_data['Close'].iloc[0])
                        end_price = float(stock_data['Close'].iloc[-1])
                        stock_return = (end_price / start_price) - 1
                        
                        # 한국 주식(숫자만) vs 미국/국제 주식(알파벳 포함) 판별
                        is_korean = ticker.isdigit()
                        if is_korean:
                            start_price_str = f"₩{start_price:,.0f}"
                            end_price_str = f"₩{end_price:,.0f}"
                        else:
                            start_price_str = f"${start_price:,.2f}"
                            end_price_str = f"${end_price:,.2f}"
                        
                        stock_performance.append({
                            "종목": ticker,
                            "수익률": f"{stock_return:.2%}",
                            "시작가": start_price_str,
                            "종료가": end_price_str
                        })
                    except:
                        pass
                
                if stock_performance:
                    stock_perf_df = pd.DataFrame(stock_performance)
                    st.dataframe(stock_perf_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                st.write("**분산 투자 효과 분석**")
                correlation_chart = plot_correlation_heatmap(portfolio_data)
                if correlation_chart:
                    st.plotly_chart(correlation_chart, use_container_width=True)
                    
                    st.info("""
                    💡 **상관관계 해석**:
                    - **1에 가까움**: 두 종목이 함께 올라가는 경향 (분산 효과 낮음)
                    - **0 근처**: 두 종목이 독립적으로 움직임 (분산 효과 있음)
                    - **-1에 가까움**: 한 종목이 올라가면 다른 종목이 내려감 (최고의 분산 효과)
                    """)
                else:
                    st.warning("상관관계 데이터를 불러올 수 없습니다.")
            
            # ===== TAB 3: AI Analysis =====
            with tab3:
                st.subheader("🤖 LLM 기반 포트폴리오 분석")
                
                if groq_client is None:
                    st.warning("⚠️ Groq API가 설정되지 않았습니다. AI 분석 기능을 사용하려면 API 키를 설정하세요.")
                else:
                    analysis_type = st.radio("분석 유형", ["일반 분석", "심화 분석"], horizontal=True)
                    
                    if st.button("🔍 분석 시작"):
                        with st.spinner("분석 중..."):
                            analysis = analyze_portfolio_with_llm(
                                portfolio_stats,
                                tickers,
                                weights,
                                analysis_type="detailed" if analysis_type == "심화 분석" else "general"
                            )
                            
                            if analysis:
                                st.success("✅ 분석 완료")
                                if analysis_type == "일반 분석":
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**전체 평가**")
                                        st.info(analysis.get("overall_assessment", "분석 실패"))
                                        
                                        st.write("**장점**")
                                        for strength in analysis.get("strengths", []):
                                            st.write(f"✅ {strength}")
                                    
                                    with col2:
                                        st.write("**약점**")
                                        for weakness in analysis.get("weaknesses", []):
                                            st.write(f"❌ {weakness}")
                                        
                                        st.write("**위험 평가**")
                                        st.warning(analysis.get("risk_assessment", "분석 실패"))
                                    
                                    st.write("**개선 제안**")
                                    for i, rec in enumerate(analysis.get("recommendations", []), 1):
                                        st.write(f"{i}. {rec}")
                                
                                else:
                                    st.write("**변동성 및 위험 분석**")
                                    st.info(analysis.get("volatility_analysis", "분석 실패"))
                                    
                                    st.write("**효율성 분석**")
                                    st.info(analysis.get("efficiency_analysis", "분석 실패"))
                                    
                                    st.write("**위험 프로필**")
                                    st.warning(analysis.get("risk_profile", "분석 실패"))
                                    
                                    st.write("**최적화 전략**")
                                    st.info(analysis.get("optimization_strategy", "분석 실패"))
                                    
                                    st.write("**타이밍 제안**")
                                    st.write(analysis.get("timing_recommendations", "분석 실패"))
            
            # ===== TAB 4: Financial Terms =====
            with tab4:
                st.subheader("💡 금융용어 학습 센터")
                
                if groq_client is None:
                    st.warning("⚠️ Groq API가 설정되지 않았습니다.")
                else:
                    # 탭으로 구분
                    subtab1, subtab2 = st.tabs(["📚 기본 정의", "❓ 질문하기"])
                    
                    with subtab1:
                        st.write("각 금융용어의 기본 정의를 확인하세요.")
                        search_term = st.text_input("📚 용어 검색", placeholder="예: 샤프지수, 변동성")
                        
                        if search_term:
                            matching_terms = [t for t in FINANCIAL_TERMS.keys() if search_term.lower() in t.lower()]
                            
                            if matching_terms:
                                for term in matching_terms:
                                    with st.expander(f"📌 **{term}**", expanded=True if len(matching_terms) == 1 else False):
                                        st.write(f"**정의**: {FINANCIAL_TERMS[term]}")
                            else:
                                st.warning(f"'{search_term}'와 일치하는 용어가 없습니다.")
                        else:
                            st.write("**전체 금융용어 목록**")
                            cols = st.columns(2)
                            for idx, (term, definition) in enumerate(FINANCIAL_TERMS.items()):
                                with cols[idx % 2]:
                                    with st.expander(f"📌 {term}"):
                                        st.write(definition)
                    
                    with subtab2:
                        st.write("용어에 대해 구체적인 질문을 하면 AI가 답변해줍니다.")
                        
                        # 용어 선택
                        selected_term = st.selectbox(
                            "질문할 용어 선택",
                            options=list(FINANCIAL_TERMS.keys()),
                            help="질문하고 싶은 금융용어를 선택하세요"
                        )
                        
                        if selected_term:
                            st.write(f"**선택된 용어**: {selected_term}")
                            st.write(f"**기본 정의**: {FINANCIAL_TERMS[selected_term]}")
                            st.markdown("---")
                            
                            # 자주 묻는 질문 예시
                            st.write("**자주 묻는 질문들:**")
                            example_questions = {
                                "샤프지수": "높은 샤프지수는 왜 좋나요?",
                                "변동성": "변동성을 어떻게 줄일 수 있나요?",
                                "최대낙폭": "최대낙폭이 중요한 이유는?",
                                "포트폴리오": "포트폴리오를 어떻게 구성하나요?",
                            }
                            
                            suggested_q = example_questions.get(selected_term, "이 용어를 더 자세히 설명해주세요")
                            
                            # 질문 입력
                            user_question = st.text_area(
                                "❓ 질문을 입력하세요",
                                value=suggested_q,
                                placeholder="예: 샤프지수가 높으면 무엇이 좋나요?",
                                height=100
                            )
                            
                            if st.button("💬 답변받기", key=f"ask_{selected_term}"):
                                if user_question.strip():
                                    with st.spinner("AI가 답변을 생성 중입니다..."):
                                        answer = ask_question_about_term_with_llm(selected_term, user_question)
                                        
                                        if answer:
                                            st.success("✅ AI 답변")
                                            st.info(answer)
                                        else:
                                            st.warning("❌ 답변 생성에 실패했습니다. 다시 시도해주세요.")
                                else:
                                    st.warning("❌ 질문을 입력해주세요.")
                            
                            # 추천 질문 버튼들
                            st.markdown("---")
                            st.write("**빠른 질문:**")
                            quick_questions = [
                                "이 용어가 투자 결정에 어떻게 도움이 되나요?",
                                "일반인도 이해하기 쉬운 예시를 들어주세요",
                                "이 지표가 높으면/낮으면 무엇을 의미하나요?",
                            ]
                            
                            for q in quick_questions:
                                if st.button(q, key=f"quick_{selected_term}_{q[:10]}"):
                                    with st.spinner("AI가 답변을 생성 중입니다..."):
                                        answer = ask_question_about_term_with_llm(selected_term, q)
                                        
                                        if answer:
                                            st.success("✅ AI 답변")
                                            st.info(answer)
            
            # ===== TAB 5: Advanced Analysis =====
            with tab5:
                st.subheader("📋 고급 분석")
                
                # 1. 벤치마크 비교
                st.write("## 1️⃣ 벤치마크 비교 분석")
                benchmark_col1, benchmark_col2 = st.columns(2)
                
                # 벤치마크 옵션 정의
                benchmark_options = {
                    "KOSPI (한국)": "^KS11",
                    "KOSDAQ (한국)": "^KQ11",
                    "S&P 500 (미국)": "^GSPC",
                    "NASDAQ (미국)": "^IXIC",
                    "Dow Jones (미국)": "^DJI"
                }
                
                with benchmark_col1:
                    selected_benchmark = st.selectbox(
                        "벤치마크 지수 선택", 
                        list(benchmark_options.keys())
                    )
                    benchmark_ticker = benchmark_options[selected_benchmark]
                
                with benchmark_col2:
                    st.caption(f"선택된 티커: {benchmark_ticker}")
                    # (Fallback 설명: 지수 데이터 누락 시 주요 ETF 데이터 사용)
                
                with st.spinner("벤치마크 데이터 로드 중..."):
                    # (이하 calculate_benchmark_comparison 호출 코드는 그대로 유지)
                    benchmark_comparison = calculate_benchmark_comparison(
                        portfolio_data["portfolio_value"],
                        portfolio_data["dates"],
                        benchmark_ticker,
                        start_date,
                        end_date
                    )

                if benchmark_comparison:
                    # 벤치마크 차트
                    benchmark_chart = plot_benchmark_comparison(benchmark_comparison)
                    if benchmark_chart:
                        st.plotly_chart(benchmark_chart, use_container_width=True)
                    
                    # 비교 지표
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "포트폴리오 수익률",
                            f"{benchmark_comparison['portfolio_cumulative']:.2%}",
                            delta=f"{benchmark_comparison['outperformance']:.2%}"
                        )
                    with col2:
                        st.metric("벤치마크 수익률", f"{benchmark_comparison['benchmark_cumulative']:.2%}")
                    with col3:
                        st.metric("베타", f"{benchmark_comparison['beta']:.3f}", 
                                 help="1보다 크면 시장보다 변동성 큼, 작으면 작음")
                    with col4:
                        st.metric("알파", f"{benchmark_comparison['alpha']:.3%}",
                                 help="벤치마크 대비 초과 수익률")
                    
                    st.info(f"""
                    💡 **벤치마크 분석 해석**:
                    - **아웃퍼포먼스**: {benchmark_comparison['outperformance']:.2%} {'✅ 벤치마크를 상회' if benchmark_comparison['outperformance'] > 0 else '❌ 벤치마크 미만'}
                    - **베타 ({benchmark_comparison['beta']:.3f}*)**: {'시장 변동성보다 큼' if benchmark_comparison['beta'] > 1 else '시장 변동성보다 작음'}
                    - **알파 ({benchmark_comparison['alpha']:.3%})**: {'초과 수익 창출' if benchmark_comparison['alpha'] > 0 else '초과 손실'}
                    """)
                else:
                    st.error(f"""
                    ❌ **벤치마크 데이터를 불러올 수 없습니다.**
                    
                    **원인:**
                    - 선택한 시간 범위에 벤치마크 데이터가 없거나
                    - 포트폴리오와 벤치마크의 거래일이 일치하지 않음
                    
                    **해결 방법:**
                    - 시간 범위를 더 최근으로 조정해보세요
                    - 다른 벤치마크를 선택해보세요
                    """)
                
                
                st.markdown("---")
                
                # 2. 리밸런싱 분석
                st.write("## 2️⃣ 정기 리밸런싱 효과")
                rebal_col1, rebal_col2 = st.columns(2)
                with rebal_col1:
                    rebalance_freq = st.select_slider("리밸런싱 빈도", 
                                                     options=["month", "quarter", "year"],
                                                     value="quarter",
                                                     format_func=lambda x: 
                                                     "월별" if x == "month" else 
                                                     "분기별" if x == "quarter" else "연간")
                with rebal_col2:
                    st.empty()
                
                rebalance_data = calculate_rebalancing_effect(
                    portfolio_data, tickers, weights, rebalance_freq
                )
                
                if rebalance_data:
                    rebal_chart = plot_rebalancing_comparison(rebalance_data)
                    if rebal_chart:
                        st.plotly_chart(rebal_chart, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("리밸런싱 수익률", f"{rebalance_data['rebalanced_cumulative']:.2%}")
                    with col2:
                        st.metric("미적용 수익률", f"{rebalance_data['original_cumulative']:.2%}")
                    with col3:
                        st.metric("차이", f"{rebalance_data['difference']:+.2%}")
                    
                    if rebalance_data['difference'] > 0:
                        st.success(f"✅ 리밸런싱이 {abs(rebalance_data['difference']):.2%} 더 유리했습니다!")
                    else:
                        st.info(f"ℹ️ 리밸런싱 미적용이 {abs(rebalance_data['difference']):.2%} 더 수익성이 높았습니다.")
                else:
                    st.warning("리밸런싱 데이터를 계산할 수 없습니다.")
                
                st.markdown("---")
                
                # 3. 월별 성과 분석
                st.write("## 3️⃣ 월별 수익률 분석")
                monthly_chart = plot_monthly_heatmap(portfolio_data["portfolio_value"])
                
                if monthly_chart:
                    st.plotly_chart(monthly_chart, use_container_width=True)
                    st.info("""
                    💡 **월별 분석 팁**:
                    - 빨강: 음수 수익률 (손실)
                    - 노랑/초록: 양수 수익률 (수익)
                    - 각 셀의 숫자는 해당 월의 수익률(%)
                    """)
                else:
                    st.warning("월별 데이터를 계산할 수 없습니다.")

else:
    st.info("👈 왼쪽 사이드바에서 포트폴리오를 설정하세요.")
