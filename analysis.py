import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# [설정 영역] 하드코딩 방지 원칙 적용
# 추후 다른 종목이나 기간을 분석할 때 이 부분만 수정하면 되도록 분리합니다.
# =====================================================================
CONFIG = {
    "TICKER": "KC=F",        # 분석 대상: 국제 커피 원두 선물 티커
    "PERIOD": "5y",          # 분석 기간: 장기 사이클을 확인하기 위한 최근 5년
    "MA_WINDOW": 60,         # 이동평균(Trend) 기준: 농산물 특성을 고려한 60일(약 1분기)
    "OUTPUT_DIR": "./images" # 시각화 이미지가 저장될 디렉토리 경로
}

def setup_environment():
    """
    [환경 설정] 이미지를 저장할 폴더가 없으면 자동으로 생성합니다.
    """
    if not os.path.exists(CONFIG["OUTPUT_DIR"]):
        os.makedirs(CONFIG["OUTPUT_DIR"])

def fetch_and_inspect_data():
    """
    [데이터 수집 및 요구사항 확인] 
    yfinance API를 사용해 데이터를 다운로드하고 기본 정보를 출력합니다.
    """
    print(f"[{CONFIG['TICKER']}] 데이터를 수집 중입니다... (기간: {CONFIG['PERIOD']})")
    
    # 1. API를 통한 시계열 데이터 수집
    df = yf.download(CONFIG["TICKER"], period=CONFIG["PERIOD"])
    
    # 2. 에러 방지 (Fail-Fast): 데이터가 비어있으면 조용히 넘기지 않고 프로그램을 강제 종료시킵니다.
    if df.empty:
        raise ValueError("데이터를 불러오지 못했습니다. 네트워크 상태나 Ticker를 확인하세요.")
        
    # 3. 호환성 패치: 최신 yfinance의 튜플형 컬럼 구조를 단일 문자열로 평탄화합니다.
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # 4. 데이터 팩트 출력
    print("\n[1. 데이터 기본 정보 확인]")
    print(f" - 데이터 기간: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")
    print(f" - 총 데이터 포인트: {len(df)}개") 
    print(f" - 보유 컬럼: {', '.join(df.columns)}")
    
    return df

def clean_data(df):
    """
    [데이터 정제] 주말 및 공휴일 휴장으로 인한 결측치(NaN)를 처리합니다.
    평균값이 아닌 마지막 거래일의 가격이 유지된다는 가정하에 Forward Fill(ffill)을 적용합니다.
    """
    missing_count = df.isnull().sum().sum()
    print(f" - 초기 결측치 개수: {missing_count}개")
    
    if missing_count > 0:
        df = df.ffill() 
        print(" - 결측치 처리 완료 (Forward Fill 적용)")
    print("-" * 50)
    
    return df

def analyze_and_extract_facts(df):
    """
    [시계열 분석 로직 및 팩트 추출]
    리포트 작성에 필요한 수치(트렌드, 노이즈, 이격도, 통계표)를 수학적으로 계산합니다.
    """
    # 1. 시계열 분석 핵심 파생 변수 생성
    df['MA'] = df['Close'].rolling(window=CONFIG["MA_WINDOW"]).mean() # 추세선
    df['Daily_Return'] = df['Close'].pct_change() * 100               # 일일 수익률(노이즈)
    df['Disparity'] = (df['Close'] / df['MA']) * 100                  # 이격도(과열 판단)
    
    # 2. 계절성 분석을 위한 연도/월 변수 생성
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    
    # 초기 이동평균선 계산을 위해 발생하는 NaN 제거
    df_valid = df.dropna(subset=['Daily_Return', 'Disparity'])
    
    print("\n[2. 데이터 관찰 팩트 추출]")
    
    # 변동성 극단값 추출
    max_up = df_valid.loc[df_valid['Daily_Return'].idxmax()]
    max_down = df_valid.loc[df_valid['Daily_Return'].idxmin()]
    print(f" - [변동성] 최대 상승일: {max_up.name.strftime('%Y-%m-%d')} (+{max_up['Daily_Return']:.2f}%)")
    print(f" - [변동성] 최대 하락일: {max_down.name.strftime('%Y-%m-%d')} ({max_down['Daily_Return']:.2f}%)")
    
    # 이격도 극단값(과열일) 추출
    max_disp = df_valid.loc[df_valid['Disparity'].idxmax()]
    print(f" - [이격도] 최대 과열일: {max_disp.name.strftime('%Y-%m-%d')} (이격도 {max_disp['Disparity']:.2f})")
    print("-" * 50)
    
    # 3. 연도별 통계 요약표 마크다운 안전 출력 (별도 라이브러리 불필요)
    print("\n[3. 연도별 변동성 통계 요약 (리포트 삽입용)]")
    yearly_stats = df_valid.groupby('Year')['Daily_Return'].agg(['mean', 'std', 'min', 'max']).round(2)
    
    print("| Year | 평균 수익률(%) | 표준편차(리스크) | 최대 하락(%) | 최대 상승(%) |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for year, row in yearly_stats.iterrows():
        print(f"| {year} | {row['mean']} | {row['std']} | {row['min']} | {row['max']} |")
    print("-" * 50)
    
    return df

def visualize_results(df):
    """
    [시각화] 데이터의 가독성을 높이기 위한 단순함 우선 원칙이 적용된 시각화 함수입니다.
    총 4개의 차트(트렌드, 노이즈, 이격도, 박스플롯)를 생성합니다.
    """
    sns.set_theme(style="whitegrid")
    output = CONFIG["OUTPUT_DIR"]
    ticker = CONFIG["TICKER"]
    window = CONFIG["MA_WINDOW"]
    
    # [차트 1] 트렌드 분석: 종가 vs 이동평균선
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df['Close'], label='Daily Close', color='#4B5563', alpha=0.5, linewidth=1)
    plt.plot(df.index, df['MA'], label=f'{window}-Day MA', color='#2563EB', linewidth=2)
    plt.title(f"{ticker} Price Trend", fontsize=14, fontweight='bold')
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output}/01_trend.png")
    plt.close()

    # [차트 2] 변동성 분석: 일일 수익률
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['Daily_Return'], label='Daily Return (%)', color='#DC2626', linewidth=1)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title(f"{ticker} Volatility (Noise)", fontsize=14, fontweight='bold')
    plt.ylabel("Return (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output}/02_volatility.png")
    plt.close()

    # [차트 3] 이격도 분석: 과열 및 평균 회귀 파악
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
    
    # [차트 4] 월별 수익률 분포 (박스 플롯): 계절성 파악
    plt.figure(figsize=(12, 5))
    sns.boxplot(x='Month', y='Daily_Return', data=df, color='#93C5FD', flierprops={"marker": "x", "color": "#DC2626"})
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title(f"{ticker} Monthly Return Distribution (Seasonality & Outliers)", fontsize=14, fontweight='bold')
    plt.ylabel("Daily Return (%)")
    plt.xlabel("Month")
    plt.tight_layout()
    plt.savefig(f"{output}/04_seasonality_boxplot.png")
    plt.close()
    
    print("\n[4. 시각화 완료] 4개의 차트가 성공적으로 저장되었습니다.")

def main():
    """
    [메인 파이프라인] 전체 데이터 흐름 제어
    """
    setup_environment()
    try:
        df_raw = fetch_and_inspect_data()
        df_clean = clean_data(df_raw)
        df_analyzed = analyze_and_extract_facts(df_clean)
        visualize_results(df_analyzed)
        
    except Exception as e:
        print(f"\n❌ 에러 발생 (실행 중단): {e}")
        raise

if __name__ == "__main__":
    main()