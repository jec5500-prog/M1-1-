import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 기획 원칙: 하드코딩 방지를 위한 설정 딕셔너리
CONFIG = {
    "TICKER": "KC=F",        # 커피 원두 선물
    "PERIOD": "5y",          # 5년 데이터
    "MA_WINDOW": 60,         # 60일(약 3개월) 이동평균선 기준
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

def analyze_disparity(df, ma_window):
    """이동평균선 대비 실제 주가의 이격도 계산"""
    # 1. 이동평균 계산
    df[f'MA_{ma_window}'] = df['Close'].rolling(window=ma_window).mean()
    
    # 2. 이격도(Disparity) 계산: (현재가 / 이동평균선) * 100
    # 100이면 이동평균선과 일치, 110이면 평균보다 10% 과열 상태를 의미
    df['Disparity'] = (df['Close'] / df[f'MA_{ma_window}']) * 100
    
    # 초기 NaN 값 제거 (MA 계산 시 발생)
    df_valid = df.dropna(subset=['Disparity'])
    
    # 가장 과열/침체되었던 시점 추출 (팩트 도출)
    max_disp_date = df_valid['Disparity'].idxmax()
    max_disp_value = df_valid['Disparity'].max()
    
    min_disp_date = df_valid['Disparity'].idxmin()
    min_disp_value = df_valid['Disparity'].min()
    
    print("\n📊 [데이터 관찰 팩트: 이격도 극단값]")
    print(f" - 최대 과열일(가격 팽창): {max_disp_date.strftime('%Y-%m-%d')} (이격도 {max_disp_value:.2f})")
    print(f" - 최대 침체일(가격 수축): {min_disp_date.strftime('%Y-%m-%d')} (이격도 {min_disp_value:.2f})")
    print("-" * 50)
    
    return df

def plot_disparity(df, ticker, ma_window, output_dir):
    """세 번째 그래프: 이격도 분석 시각화"""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 4))
    
    # 이격도 라인 (Primary Color)
    plt.plot(df.index, df['Disparity'], label=f'Disparity ({ma_window}-Day MA)', color='#2563EB', linewidth=1.5)
    
    # 기준선 (100 = 이동평균과 동일한 지점)
    plt.axhline(100, color='black', linewidth=1.5, linestyle='-', label='Baseline (100)')
    
    # 과열/침체 기준선 (가이드라인 용도: 통상 110 이상, 90 이하)
    plt.axhline(110, color='#DC2626', linewidth=1, linestyle='--', alpha=0.7, label='Overbought (110)')
    plt.axhline(90, color='#DC2626', linewidth=1, linestyle='--', alpha=0.7, label='Oversold (90)')
    
    plt.title(f"{ticker} Price Disparity Analysis (Mean Reversion)", fontsize=15, fontweight='bold')
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Disparity (%)", fontsize=11)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    save_path = f"{output_dir}/03_coffee_disparity.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 시각화 완료: {save_path} 파일이 생성되었습니다.")

def main():
    setup_environment(CONFIG["OUTPUT_DIR"])
    
    try:
        raw_df = fetch_data(CONFIG["TICKER"], CONFIG["PERIOD"])
        cleaned_df = clean_data(raw_df)
        analyzed_df = analyze_disparity(cleaned_df, CONFIG["MA_WINDOW"])
        plot_disparity(analyzed_df, CONFIG["TICKER"], CONFIG["MA_WINDOW"], CONFIG["OUTPUT_DIR"])
        
    except Exception as e:
        print(f"❌ 분석 중 에러가 발생했습니다: {e}")
        raise

if __name__ == "__main__":
    main()