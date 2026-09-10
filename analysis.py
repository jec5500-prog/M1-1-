import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# [기획 원칙 1] 하드코딩 금지: 모든 환경 및 파라미터 변수는 최상단에서 관리
CONFIG = {
    "TICKER": "KC=F",        
    "PERIOD": "5y",          
    "MA_WINDOW": 60,         
    "OUTPUT_DIR": "./images"
}

def setup_environment():
    """디렉토리 환경 셋업"""
    if not os.path.exists(CONFIG["OUTPUT_DIR"]):
        os.makedirs(CONFIG["OUTPUT_DIR"])

def fetch_and_inspect_data():
    """데이터 수집 및 기본 정보(요구사항) 확인"""
    print(f"[{CONFIG['TICKER']}] 데이터를 수집 중입니다... (기간: {CONFIG['PERIOD']})")
    df = yf.download(CONFIG["TICKER"], period=CONFIG["PERIOD"])
    
    # [기획 원칙 4] 에러는 삼키지 않는다.
    if df.empty:
        raise ValueError("데이터를 불러오지 못했습니다. 네트워크 상태나 Ticker를 확인하세요.")
        
    # 최신 yfinance 라이브러리의 튜플형 컬럼 구조를 문자열로 평탄화 (호환성 패치)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # [요구사항 충족] 데이터 기본 정보(기간, 컬럼) 터미널 출력
    print("\n[1. 데이터 기본 정보 확인]")
    print(f" - 데이터 기간: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")
    print(f" - 총 데이터 포인트: {len(df)}개")
    print(f" - 보유 컬럼: {', '.join(df.columns)}")
    return df

def clean_data(df):
    """결측치 확인 및 처리"""
    missing_count = df.isnull().sum().sum()
    print(f" - 초기 결측치 개수: {missing_count}개")
    if missing_count > 0:
        df = df.ffill()
        print(" - 결측치 처리 완료 (Forward Fill 적용)")
    print("-" * 50)
    return df

def analyze_and_extract_facts(df):
    """3가지 시계열 분석 기법 적용 및 팩트 추출"""
    # 1. 트렌드 분석 (이동평균)
    df['MA'] = df['Close'].rolling(window=CONFIG["MA_WINDOW"]).mean()
    
    # 2. 노이즈 분석 (일일 변화율)
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    # 3. 평균 회귀 분석 (이격도)
    df['Disparity'] = (df['Close'] / df['MA']) * 100
    
    # --- 데이터 관찰(Fact) 추출 ---
    df_valid = df.dropna(subset=['Daily_Return', 'Disparity'])
    print("\n[2. 데이터 관찰 팩트 추출]")
    
    max_up = df_valid.loc[df_valid['Daily_Return'].idxmax()]
    max_down = df_valid.loc[df_valid['Daily_Return'].idxmin()]
    print(f" - [변동성] 최대 상승일: {max_up.name.strftime('%Y-%m-%d')} (+{max_up['Daily_Return']:.2f}%)")
    print(f" - [변동성] 최대 하락일: {max_down.name.strftime('%Y-%m-%d')} ({max_down['Daily_Return']:.2f}%)")
    
    max_disp = df_valid.loc[df_valid['Disparity'].idxmax()]
    print(f" - [이격도] 최대 과열일: {max_disp.name.strftime('%Y-%m-%d')} (이격도 {max_disp['Disparity']:.2f})")
    print("-" * 50)
    
    return df

def visualize_results(df):
    """3가지 시각화 결과물 생성 (디자인 시스템 적용)"""
    sns.set_theme(style="whitegrid")
    output = CONFIG["OUTPUT_DIR"]
    ticker = CONFIG["TICKER"]
    window = CONFIG["MA_WINDOW"]
    
    # [차트 1] 트렌드 분석 (Primary Color)
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df['Close'], label='Daily Close', color='#4B5563', alpha=0.5, linewidth=1)
    plt.plot(df.index, df['MA'], label=f'{window}-Day MA', color='#2563EB', linewidth=2)
    plt.title(f"{ticker} Price Trend", fontsize=14, fontweight='bold')
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output}/01_trend.png")
    plt.close()

    # [차트 2] 변동성 분석 (Alert Color)
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['Daily_Return'], label='Daily Return (%)', color='#DC2626', linewidth=1)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title(f"{ticker} Volatility (Noise)", fontsize=14, fontweight='bold')
    plt.ylabel("Return (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output}/02_volatility.png")
    plt.close()

    # [차트 3] 이격도 분석
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['Disparity'], label='Disparity', color='#2563EB', linewidth=1.5)
    plt.axhline(100, color='black', linewidth=1.5, linestyle='-')
    plt.axhline(110, color='#DC2626', linewidth=1, linestyle='--', alpha=0.7, label='Overbought (110)')
    plt.axhline(90, color='#DC2626', linewidth=1, linestyle='--', alpha=0.7, label='Oversold (90)')
    plt.title(f"{ticker} Disparity (Mean Reversion)", fontsize=14, fontweight='bold')
    plt.ylabel("Disparity (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output}/03_disparity.png")
    plt.close()
    
    print("\n[3. 시각화 완료] 3개의 차트가 성공적으로 저장되었습니다.")

def main():
    setup_environment()
    try:
        # 단일 파이프라인으로 순차적 실행 (재현성 확보)
        df_raw = fetch_and_inspect_data()
        df_clean = clean_data(df_raw)
        df_analyzed = analyze_and_extract_facts(df_clean)
        visualize_results(df_analyzed)
    except Exception as e:
        print(f"\n❌ 에러 발생 (실행 중단): {e}")
        raise

if __name__ == "__main__":
    main()