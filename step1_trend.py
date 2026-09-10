import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 기획 원칙: 하드코딩 방지를 위한 설정 딕셔너리 분리
CONFIG = {
    "TICKER": "KC=F",        # 커피 원두 선물
    "PERIOD": "5y",          # 계절성과 장기 트렌드를 보기 위한 5년 데이터
    "MA_WINDOW": 60,         # 커피 농작물의 특성을 고려한 중장기 추세선 (약 3개월)
    "OUTPUT_DIR": "./images"
}

def setup_environment(output_dir):
    """결과물을 저장할 디렉토리 생성"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def fetch_data(ticker, period):
    """Yahoo Finance 데이터 수집 (에러 삼키지 않음)"""
    print(f"[{ticker}] 데이터를 수집 중입니다... (기간: {period})")
    df = yf.download(ticker, period=period)
    
    if df.empty:
        raise ValueError("데이터를 불러오지 못했습니다. Ticker나 네트워크 상태를 확인하세요.")
    return df

def clean_data(df):
    """결측치 확인 및 정제 (Forward Fill)"""
    missing_count = df.isnull().sum().sum()
    print(f"초기 결측치 개수: {missing_count}개")
    if missing_count > 0:
        df = df.ffill()
    return df

def analyze_trend(df, ma_window):
    """트렌드 분석을 위한 이동평균 계산"""
    df[f'MA_{ma_window}'] = df['Close'].rolling(window=ma_window).mean()
    return df

def plot_first_graph(df, ticker, ma_window, output_dir):
    """첫 번째 그래프: 종가 트렌드와 이동평균선 시각화 (디자인 시스템 적용)"""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 6))
    
    # 관찰 가능한 노이즈(일일 종가)와 해석을 위한 트렌드(이동평균) 분리 시각화
    plt.plot(df.index, df['Close'], label='Daily Close Price', color='#4B5563', alpha=0.5, linewidth=1)
    plt.plot(df.index, df[f'MA_{ma_window}'], label=f'{ma_window}-Day Moving Average', color='#2563EB', linewidth=2)
    
    plt.title(f"{ticker} Coffee Futures Price Trend ({CONFIG['PERIOD']})", fontsize=15, fontweight='bold')
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Price (US Cents per Pound)", fontsize=11)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    save_path = f"{output_dir}/01_coffee_trend.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 시각화 완료: {save_path} 파일이 생성되었습니다.")

def main():
    setup_environment(CONFIG["OUTPUT_DIR"])
    
    try:
        raw_df = fetch_data(CONFIG["TICKER"], CONFIG["PERIOD"])
        cleaned_df = clean_data(raw_df)
        analyzed_df = analyze_trend(cleaned_df, CONFIG["MA_WINDOW"])
        plot_first_graph(analyzed_df, CONFIG["TICKER"], CONFIG["MA_WINDOW"], CONFIG["OUTPUT_DIR"])
        
    except Exception as e:
        print(f"❌ 분석 중 에러가 발생했습니다: {e}")
        raise  # 명시적 에러 발생 (기획서 원칙)

if __name__ == "__main__":
    main()