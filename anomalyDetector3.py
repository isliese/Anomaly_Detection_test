import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

class CryptoAnomalyDetector:
    def __init__(self, data, coin_name):
        self.data = data.copy()
        self.coin_name = coin_name
        self.prepare_data()
        
    def prepare_data(self):
        """데이터 전처리 및 특성 엔지니어링"""
        # timestamp를 datetime으로 변환 (UTC 기준)
        self.data['datetime'] = pd.to_datetime(self.data['timestamp'], unit='ms', utc=True)
        
        # 기본 가격 변동률 계산
        self.data['price_change'] = self.data['close'].pct_change()
        self.data['price_change_abs'] = abs(self.data['price_change'])
        
        # 가격 변동성 (high-low spread)
        self.data['hl_spread'] = (self.data['high'] - self.data['low']) / self.data['close']
        
        # 거래량 변화율
        self.data['volume_change'] = self.data['volume'].pct_change()
        self.data['volume_change_abs'] = abs(self.data['volume_change'])
        
        # 가격-거래량 곱 (거래대금)
        self.data['turnover'] = self.data['close'] * self.data['volume']
        self.data['turnover_change'] = self.data['turnover'].pct_change()
        
        # 롤링 통계량 계산 (5분, 15분, 30분 윈도우)
        for window in [5, 15, 30]:
            self.data[f'price_volatility_{window}m'] = self.data['price_change'].rolling(window).std()
            self.data[f'volume_ma_{window}m'] = self.data['volume'].rolling(window).mean()
            self.data[f'volume_std_{window}m'] = self.data['volume'].rolling(window).std()
            
        # NaN 값 제거
        self.data = self.data.dropna().reset_index(drop=True)
    
    def method1_statistical_outliers(self, columns=['price_change_abs', 'volume_change_abs'], 
                                   threshold_method='iqr', z_threshold=3, iqr_multiplier=1.5):
        """방법 1: 통계적 이상치 탐지 (Z-score, IQR)"""
        print("=" * 60)
        print(f"방법 1: 통계적 이상치 탐지 - {self.coin_name}")
        print("=" * 60)
        
        results = {}
        total_outliers = 0
        
        for col in columns:
            if col not in self.data.columns:
                continue
                
            if threshold_method == 'zscore':
                # Z-score 방법
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outliers = z_scores > z_threshold
                threshold_value = z_threshold
                method_name = f"Z-score (임계값: {z_threshold})"
                
            elif threshold_method == 'iqr':
                # IQR 방법
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                outliers = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                threshold_value = iqr_multiplier
                method_name = f"IQR (배수: {iqr_multiplier})"
            
            outlier_count = outliers.sum()
            outlier_percentage = (outlier_count / len(self.data)) * 100
            total_outliers += outlier_count
            
            results[col] = {
                'method': method_name,
                'outliers': outliers,
                'count': outlier_count,
                'percentage': outlier_percentage,
                'outlier_indices': self.data[outliers].index.tolist()
            }
            
            print(f"\n📊 {col} 분석 결과 ({method_name}):")
            print(f"   - 이상치 개수: {outlier_count}개 ({outlier_percentage:.2f}%)")
            print(f"   - 평균: {self.data[col].mean():.6f}")
            print(f"   - 표준편차: {self.data[col].std():.6f}")
            
            if outlier_count > 0:
                outlier_data = self.data[outliers]
                print(f"   - 이상치 최대값: {self.data[col][outliers].max():.6f}")
                print(f"   - 이상치 발생 시간 (최근 5개):")
                for idx in outlier_data.index[-5:]:
                    print(f"     {self.data.loc[idx, 'datetime'].strftime('%Y-%m-%d %H:%M:%S')} - 값: {self.data.loc[idx, col]:.6f}")
        
        results['total_outliers'] = total_outliers
        return results
    
    def method2_rolling_deviation(self, windows=[5, 15, 30], threshold_multiplier=2.5):
        """방법 2: 롤링 윈도우 기반 편차 분석"""
        print("\n" + "=" * 60)
        print(f"방법 2: 롤링 윈도우 기반 편차 분석 - {self.coin_name}")
        print("=" * 60)
        
        results = {}
        total_combined_anomalies = 0
        
        for window in windows:
            print(f"\n📈 {window}분 윈도우 분석:")
            
            # 가격 변동성 이상치
            price_volatility_col = f'price_volatility_{window}m'
            if price_volatility_col in self.data.columns:
                vol_mean = self.data[price_volatility_col].mean()
                vol_std = self.data[price_volatility_col].std()
                vol_threshold = vol_mean + threshold_multiplier * vol_std
                
                price_anomalies = self.data[price_volatility_col] > vol_threshold
                price_anomaly_count = price_anomalies.sum()
                
                print(f"   가격 변동성 이상치: {price_anomaly_count}개 ({(price_anomaly_count/len(self.data)*100):.2f}%)")
                print(f"   임계값: {vol_threshold:.6f} (평균 + {threshold_multiplier}σ)")
            
            # 거래량 이상치
            volume_std_col = f'volume_std_{window}m'
            if volume_std_col in self.data.columns:
                vol_std_mean = self.data[volume_std_col].mean()
                vol_std_std = self.data[volume_std_col].std()
                vol_std_threshold = vol_std_mean + threshold_multiplier * vol_std_std
                
                volume_anomalies = self.data[volume_std_col] > vol_std_threshold
                volume_anomaly_count = volume_anomalies.sum()
                
                print(f"   거래량 변동성 이상치: {volume_anomaly_count}개 ({(volume_anomaly_count/len(self.data)*100):.2f}%)")
                print(f"   임계값: {vol_std_threshold:.2f}")
            
            # 복합 이상치 (가격 + 거래량 동시 발생)
            if price_volatility_col in self.data.columns and volume_std_col in self.data.columns:
                combined_anomalies = price_anomalies & volume_anomalies
                combined_count = combined_anomalies.sum()
                total_combined_anomalies += combined_count
                print(f"   복합 이상치 (가격+거래량): {combined_count}개 ({(combined_count/len(self.data)*100):.2f}%)")
                
                results[f'{window}m'] = {
                    'price_anomalies': price_anomalies,
                    'volume_anomalies': volume_anomalies,
                    'combined_anomalies': combined_anomalies,
                    'combined_count': combined_count
                }
        
        results['total_combined_anomalies'] = total_combined_anomalies
        return results
    
    def method3_percentile_based(self, percentile_threshold=95):
        """방법 3: 백분위수 기반 이상치 탐지"""
        print("\n" + "=" * 60)
        print(f"방법 3: 백분위수 기반 이상치 탐지 ({percentile_threshold}th percentile) - {self.coin_name}")
        print("=" * 60)
        
        features = ['price_change_abs', 'volume_change_abs', 'hl_spread', 'turnover_change']
        results = {}
        total_anomalies = 0
        
        for feature in features:
            if feature not in self.data.columns:
                continue
                
            threshold = self.data[feature].quantile(percentile_threshold / 100)
            anomalies = self.data[feature] > threshold
            anomaly_count = anomalies.sum()
            total_anomalies += anomaly_count
            
            results[feature] = {
                'threshold': threshold,
                'anomalies': anomalies,
                'count': anomaly_count,
                'percentage': (anomaly_count / len(self.data)) * 100
            }
            
            print(f"\n📊 {feature}:")
            print(f"   - {percentile_threshold}th percentile 임계값: {threshold:.6f}")
            print(f"   - 이상치 개수: {anomaly_count}개 ({(anomaly_count/len(self.data)*100):.2f}%)")
            
            if anomaly_count > 0:
                anomaly_times = self.data[anomalies]['datetime'].tail(3)
                print(f"   - 최근 이상치 발생 시간:")
                for dt in anomaly_times:
                    print(f"     {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        
        results['total_anomalies'] = total_anomalies
        return results
    
    def method4_composite_anomaly_score(self):
        """방법 4: 복합 이상치 점수 계산"""
        print("\n" + "=" * 60)
        print(f"방법 4: 복합 이상치 점수 (실시간 모델용) - {self.coin_name}")
        print("=" * 60)
        
        # 정규화된 점수 계산
        features = ['price_change_abs', 'volume_change_abs', 'hl_spread']
        scores = pd.DataFrame()
        
        for feature in features:
            if feature in self.data.columns:
                # 0-1 스케일링 (최근 30분 기준)
                rolling_min = self.data[feature].rolling(window=30, min_periods=1).min()
                rolling_max = self.data[feature].rolling(window=30, min_periods=1).max()
                
                normalized = (self.data[feature] - rolling_min) / (rolling_max - rolling_min + 1e-8)
                scores[feature] = normalized.fillna(0)
        
        # 가중 복합 점수 (가격 변동: 40%, 거래량 변동: 40%, 스프레드: 20%)
        weights = {'price_change_abs': 0.4, 'volume_change_abs': 0.4, 'hl_spread': 0.2}
        
        composite_score = pd.Series(0, index=self.data.index)
        for feature, weight in weights.items():
            if feature in scores.columns:
                composite_score += scores[feature] * weight
        
        self.data['anomaly_score'] = composite_score
        
        # 이상치 임계값 설정 (상위 5%)
        anomaly_threshold = composite_score.quantile(0.95)
        high_anomalies = composite_score > anomaly_threshold
        
        print(f"📊 복합 이상치 점수 분석:")
        print(f"   - 평균 점수: {composite_score.mean():.4f}")
        print(f"   - 최대 점수: {composite_score.max():.4f}")
        print(f"   - 이상치 임계값 (95th percentile): {anomaly_threshold:.4f}")
        print(f"   - 고위험 이상치: {high_anomalies.sum()}개 ({(high_anomalies.sum()/len(self.data)*100):.2f}%)")
        
        # 최고 점수 상위 5개 시점
        top_anomalies = self.data.nlargest(5, 'anomaly_score')
        print(f"\n🚨 최고 이상치 점수 TOP 5:")
        for idx, row in top_anomalies.iterrows():
            print(f"   {row['datetime'].strftime('%Y-%m-%d %H:%M:%S')} - 점수: {row['anomaly_score']:.4f}")
            print(f"      가격변동: {row['price_change_abs']:.4f}, 거래량변동: {row['volume_change_abs']:.4f}")
        
        return composite_score, anomaly_threshold, high_anomalies.sum()
    
    def analyze_trading_patterns(self):
        """거래 패턴 분석"""
        print("\n" + "=" * 60)
        print(f"거래 패턴 및 인사이트 분석 - {self.coin_name}")
        print("=" * 60)
        
        # 시간대별 분석
        self.data['hour'] = self.data['datetime'].dt.hour
        hourly_stats = self.data.groupby('hour').agg({
            'volume': ['mean', 'std'],
            'price_change_abs': ['mean', 'std'],
            'hl_spread': 'mean'
        }).round(4)
        
        print("\n⏰ 시간대별 거래 활동:")
        print("시간\t거래량(평균)\t가격변동(평균)\t스프레드(평균)")
        for hour in range(24):
            if hour in hourly_stats.index:
                vol_mean = hourly_stats.loc[hour, ('volume', 'mean')]
                price_mean = hourly_stats.loc[hour, ('price_change_abs', 'mean')]
                spread_mean = hourly_stats.loc[hour, ('hl_spread', 'mean')]
                print(f"{hour:02d}시\t{vol_mean:,.0f}\t\t{price_mean:.4f}\t\t{spread_mean:.4f}")
        
        # 가격-거래량 상관관계
        price_volume_corr = self.data['price_change_abs'].corr(self.data['volume_change_abs'])
        print(f"\n📈 가격변동-거래량변동 상관계수: {price_volume_corr:.4f}")
        
        # 변동성 클러스터링 (연속된 고변동성 구간)
        high_volatility = self.data['price_change_abs'] > self.data['price_change_abs'].quantile(0.9)
        volatility_clusters = []
        cluster_start = None
        
        for i, is_high_vol in enumerate(high_volatility):
            if is_high_vol and cluster_start is None:
                cluster_start = i
            elif not is_high_vol and cluster_start is not None:
                if i - cluster_start >= 3:  # 3분 이상 연속 고변동성
                    volatility_clusters.append((cluster_start, i-1))
                cluster_start = None
        
        print(f"\n🔥 변동성 클러스터 (3분 이상 연속 고변동성): {len(volatility_clusters)}개")
        for start, end in volatility_clusters[-3:]:  # 최근 3개만 출력
            start_time = self.data.loc[start, 'datetime'].strftime('%H:%M:%S')
            end_time = self.data.loc[end, 'datetime'].strftime('%H:%M:%S')
            duration = end - start + 1
            print(f"   {start_time} ~ {end_time} ({duration}분 지속)")
    
    def real_time_anomaly_model(self, lookback_window=30):
        """실시간 이상 탐지 모델 시뮬레이션"""
        print("\n" + "=" * 60)
        print(f"실시간 이상 탐지 모델 시뮬레이션 - {self.coin_name}")
        print("=" * 60)
        
        anomaly_scores = []
        anomaly_flags = []
        
        for i in range(lookback_window, len(self.data)):
            # 현재 시점
            current = self.data.iloc[i]
            
            # 과거 lookback_window 기간의 기준값 계산
            historical_data = self.data.iloc[i-lookback_window:i]
            
            # 기준 통계량
            price_vol_mean = historical_data['price_change_abs'].mean()
            price_vol_std = historical_data['price_change_abs'].std()
            volume_mean = historical_data['volume'].mean()
            volume_std = historical_data['volume'].std()
            
            # 현재 값의 이상 정도 계산
            price_anomaly_score = abs(current['price_change_abs'] - price_vol_mean) / (price_vol_std + 1e-8)
            volume_anomaly_score = abs(current['volume'] - volume_mean) / (volume_std + 1e-8)
            
            # 복합 점수 (0-1 스케일)
            composite_score = min(1.0, (price_anomaly_score * 0.6 + volume_anomaly_score * 0.4) / 5)
            
            # 이상치 플래그 (점수 > 0.7)
            is_anomaly = composite_score > 0.7
            
            anomaly_scores.append(composite_score)
            anomaly_flags.append(is_anomaly)
        
        # 결과 저장
        start_idx = lookback_window
        self.data.loc[start_idx:, 'realtime_anomaly_score'] = anomaly_scores
        self.data.loc[start_idx:, 'realtime_anomaly_flag'] = anomaly_flags
        
        # 통계
        total_anomalies = sum(anomaly_flags)
        print(f"📊 실시간 모델 성능:")
        print(f"   - 분석 대상 기간: {len(anomaly_scores)}분")
        print(f"   - 탐지된 이상치: {total_anomalies}개 ({(total_anomalies/len(anomaly_scores)*100):.2f}%)")
        print(f"   - 평균 이상치 점수: {np.mean(anomaly_scores):.4f}")
        print(f"   - 최대 이상치 점수: {max(anomaly_scores):.4f}")
        
        # 최근 이상치 5개
        recent_anomalies = self.data[self.data['realtime_anomaly_flag'] == True].tail(5)
        if len(recent_anomalies) > 0:
            print(f"\n🚨 최근 탐지된 이상치:")
            for idx, row in recent_anomalies.iterrows():
                print(f"   {row['datetime'].strftime('%H:%M:%S')} - 점수: {row['realtime_anomaly_score']:.4f}")
        
        return anomaly_scores, anomaly_flags, total_anomalies
    
    def visualize_results(self):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.coin_name} 이상 상황 탐지 분석 결과', fontsize=16, fontweight='bold')
        
        # 1. 가격 변동률 시계열
        axes[0,0].plot(self.data['datetime'], self.data['price_change_abs'], alpha=0.7, linewidth=0.8)
        axes[0,0].axhline(y=self.data['price_change_abs'].quantile(0.95), color='red', 
                        linestyle='--', label='95th percentile')
        axes[0,0].set_title('가격 변동률 (절댓값)')
        axes[0,0].set_ylabel('변동률')
        axes[0,0].legend()
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. 거래량 시계열
        axes[0,1].plot(self.data['datetime'], self.data['volume'], alpha=0.7, linewidth=0.8, color='green')
        axes[0,1].axhline(y=self.data['volume'].quantile(0.95), color='red', 
                        linestyle='--', label='95th percentile')
        axes[0,1].set_title('거래량')
        axes[0,1].set_ylabel('거래량')
        axes[0,1].legend()
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. 복합 이상치 점수
        if 'anomaly_score' in self.data.columns:
            axes[1,0].plot(self.data['datetime'], self.data['anomaly_score'], 
                        alpha=0.8, linewidth=1, color='purple')
            axes[1,0].axhline(y=0.7, color='red', linestyle='--', label='이상치 임계값 (0.7)')
            axes[1,0].set_title('복합 이상치 점수')
            axes[1,0].set_ylabel('점수')
            axes[1,0].legend()
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. 가격-거래량 산점도
        scatter_sample = self.data.sample(min(500, len(self.data)))  # 샘플링으로 시각화 개선
        scatter = axes[1,1].scatter(scatter_sample['price_change_abs'], 
                                scatter_sample['volume_change_abs'],
                                alpha=0.6, s=20)
        axes[1,1].set_xlabel('가격 변동률 (절댓값)')
        axes[1,1].set_ylabel('거래량 변동률 (절댓값)')
        axes[1,1].set_title('가격 vs 거래량 변동 관계')
        
        plt.tight_layout()
        plt.show()
        
        # 추가 분석 차트
        self.plot_anomaly_distribution()

    def plot_anomaly_distribution(self):
        """이상치 분포 분석 차트"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 이상치 점수 분포
        if 'anomaly_score' in self.data.columns:
            axes[0].hist(self.data['anomaly_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0].axvline(x=0.7, color='red', linestyle='--', label='임계값 (0.7)')
            axes[0].set_title('이상치 점수 분포')
            axes[0].set_xlabel('점수')
            axes[0].set_ylabel('빈도')
            axes[0].legend()
        
        # 시간대별 이상치 발생 빈도
        if 'realtime_anomaly_flag' in self.data.columns:
            hourly_anomalies = self.data.groupby('hour')['realtime_anomaly_flag'].sum()
            axes[1].bar(hourly_anomalies.index, hourly_anomalies.values, alpha=0.7, color='coral')
            axes[1].set_title('시간대별 이상치 발생 빈도')
            axes[1].set_xlabel('시간 (UTC)')
            axes[1].set_ylabel('이상치 발생 횟수')
            axes[1].set_xticks(range(0, 24, 2))
        
        plt.tight_layout()
        plt.show()


def print_ranking_summary(method_results):
    """각 방법별 이상치 탐지 순위 출력"""
    print("\n" + "=" * 80)
    print("🏆 이상치 탐지 방법별 코인 순위 (이상치 많이 발견된 순)")
    print("=" * 80)
    
    for method_name, results in method_results.items():
        print(f"\n📈 {method_name}")
        print("-" * 50)
        
        # 결과를 이상치 개수로 정렬
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (coin, count) in enumerate(sorted_results, 1):
            print(f"   {rank}위: {coin:8s} - {count:4d}개 이상치")


# 메인 실행 부분
if __name__ == "__main__":
    # CSV 파일 경로
    file_path_dic = {
        r"candle_data\candle_data\ADA.csv": "ADA", 
        r"candle_data\candle_data\BERA.csv": "BERA", 
        r"candle_data\candle_data\ETH.csv": "ETH", 
        r"candle_data\candle_data\PENGU.csv": "PENGU",
        r"candle_data\candle_data\SOON.csv": "SOON", 
        r"candle_data\candle_data\SUNDOG.csv": "SUNDOG", 
        r"candle_data\candle_data\XRP.csv": "XRP", 
        r"candle_data\candle_data\YFI.csv": "YFI"
    }

    # 각 방법별 결과 저장용 딕셔너리
    method1_results = {}  # 통계적 이상치
    method2_results = {}  # 롤링 윈도우
    method3_results = {}  # 백분위수
    method4_results = {}  # 복합 점수
    realtime_results = {}  # 실시간 모델

    # 각 파일별로 분석 실행 
    for file_path, coin_name in file_path_dic.items():
        print(f"\n{'='*80}")
        print(f"🔍 {coin_name} 분석 시작")
        print(f"{'='*80}")
        
        try:
            # CSV 파일 읽기
            df = pd.read_csv(file_path)
            
            # 필요한 컬럼 선택 및 이름 정리
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # CryptoAnomalyDetector 초기화
            detector = CryptoAnomalyDetector(df, coin_name)
            
            # 방법 1: 통계적 이상치 탐지
            result1 = detector.method1_statistical_outliers()
            method1_results[coin_name] = result1.get('total_outliers', 0)
            
            # 방법 2: 롤링 윈도우 기반 편차 분석
            result2 = detector.method2_rolling_deviation()
            method2_results[coin_name] = result2.get('total_combined_anomalies', 0)
            
            # 방법 3: 백분위수 기반 이상치 탐지
            result3 = detector.method3_percentile_based()
            method3_results[coin_name] = result3.get('total_anomalies', 0)
            
            # 방법 4: 복합 이상치 점수 계산
            composite_score, anomaly_threshold, high_anomaly_count = detector.method4_composite_anomaly_score()
            method4_results[coin_name] = high_anomaly_count
            
            # 거래 패턴 분석
            detector.analyze_trading_patterns()
            
            # 실시간 이상치 모델 시뮬레이션
            anomaly_scores, anomaly_flags, realtime_anomaly_count = detector.real_time_anomaly_model()
            realtime_results[coin_name] = realtime_anomaly_count
            
            # 시각화 (선택적으로 주석 해제)
            # detector.visualize_results()
            
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            method1_results[coin_name] = 0
            method2_results[coin_name] = 0
            method3_results[coin_name] = 0
            method4_results[coin_name] = 0
            realtime_results[coin_name] = 0
            continue
        except Exception as e:
            print(f"❌ {coin_name} 분석 중 오류 발생: {str(e)}")
            method1_results[coin_name] = 0
            method2_results[coin_name] = 0
            method3_results[coin_name] = 0
            method4_results[coin_name] = 0
            realtime_results[coin_name] = 0
            continue

    # 전체 결과 순위 출력
    all_method_results = {
        "방법 1: 통계적 이상치 탐지 (IQR)": method1_results,
        "방법 2: 롤링 윈도우 기반 편차 분석": method2_results,
        "방법 3: 백분위수 기반 이상치 탐지": method3_results,
        "방법 4: 복합 이상치 점수": method4_results,
        "실시간 이상치 모델": realtime_results
    }
    
    print_ranking_summary(all_method_results)
    
    # 종합 순위 (모든 방법의 평균 순위)
    print("\n" + "=" * 80)
    print("🎯 종합 순위 (모든 방법 평균)")
    print("=" * 80)
    
    coin_total_scores = {}
    for coin_name in file_path_dic.values():
        total_score = (
            method1_results.get(coin_name, 0) + 
            method2_results.get(coin_name, 0) + 
            method3_results.get(coin_name, 0) + 
            method4_results.get(coin_name, 0) + 
            realtime_results.get(coin_name, 0)
        )
        coin_total_scores[coin_name] = total_score
    
    sorted_total = sorted(coin_total_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n📊 전체 이상치 종합 순위:")
    print("-" * 50)
    for rank, (coin, total_count) in enumerate(sorted_total, 1):
        print(f"   {rank}위: {coin:8s} - 총 {total_count:4d}개 이상치")
        print(f"         (방법1: {method1_results.get(coin, 0):3d}, 방법2: {method2_results.get(coin, 0):3d}, 방법3: {method3_results.get(coin, 0):3d}, 방법4: {method4_results.get(coin, 0):3d}, 실시간: {realtime_results.get(coin, 0):3d})")

    print("\n" + "=" * 80)
    print("✅ 모든 분석이 완료되었습니다!")
    print("=" * 80)