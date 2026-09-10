import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 기획 원칙: 하드코딩 방지를 위한 설정 딕셔너리
CONFIG = {
    "TICKER": "KC=F",        # 커피 원두 선물
    "PERIOD": "5y",          # 5년 데이터
    "OUTPUT_DIR": "./images"
}

def setup_environment(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def fetch_data(ticker, period):
    print(f"[{ticker}] 데이터를 수집 중입니다... (기간: {period})")
    df = yf.download(ticker, period=period)
    if df.empty:
        raise ValueError("데이터를 불러오지 못했습니다. Ticker나 네트워크 상태를 확인하세요.")
    return df

def clean_data(df):
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        df = df.ffill()
    return df

def analyze_volatility(df):
    """일일 변화율(변동성) 계산 및 극단값 탐색"""
    # 전일 대비 종가 변화율(%) 계산 (노이즈 측정)
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    # NaN 값(첫 번째 행) 제거
    df_valid = df.dropna(subset=['Daily_Return'])
    
    # 변동성이 가장 심했던 날짜(Index)와 수치 추출
    max_up_date = df_valid['Daily_Return'].idxmax()
    max_up_value = df_valid['Daily_Return'].max()
    
    max_down_date = df_valid['Daily_Return'].idxmin()
    max_down_value = df_valid['Daily_Return'].min()
    
    print("\n📊 [데이터 관찰 팩트]")
    # yfinance 데이터 구조상 date 추출 방식 적용
    print(f" - 최대 상승일: {max_up_date.strftime('%Y-%m-%d')} (+{max_up_value:.2f}%)")
    print(f" - 최대 하락일: {max_down_date.strftime('%Y-%m-%d')} ({max_down_value:.2f}%)")
    print("-" * 40)
    
    return df

def plot_volatility(df, ticker, output_dir):
    """두 번째 그래프: 일일 변화율(노이즈) 시각화"""
    sns.set_theme(style="whitegrid")
    
    # 트렌드 차트보다 세로폭을 좁게 하여 '흐름'보다 '폭발력'에 집중하도록 UI 구성
    plt.figure(figsize=(14, 4))
    
    # 기획서 디자인 시스템 적용: Alert Color (#DC2626)
    plt.plot(df.index, df['Daily_Return'], label='Daily Return (%)', color='#DC2626', linewidth=1)
    
    # 기준점(0%) 선 추가
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    plt.title(f"{ticker} Daily Return Volatility (Noise Analysis)", fontsize=15, fontweight='bold')
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Daily Return (%)", fontsize=11)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    save_path = f"{output_dir}/02_coffee_volatility.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 시각화 완료: {save_path} 파일이 생성되었습니다.")

def main():
    setup_environment(CONFIG["OUTPUT_DIR"])
    
    try:
        raw_df = fetch_data(CONFIG["TICKER"], CONFIG["PERIOD"])
        cleaned_df = clean_data(raw_df)
        
        # 변동성 분석 및 시각화 파이프라인
        analyzed_df = analyze_volatility(cleaned_df)
        plot_volatility(analyzed_df, CONFIG["TICKER"], CONFIG["OUTPUT_DIR"])
        
    except Exception as e:
        print(f"❌ 분석 중 에러가 발생했습니다: {e}")
        raise

if __name__ == "__main__":
    main()