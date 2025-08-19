import pandas as pd

# 파일 불러오기
for i in ["candle_data\candle_data\ADA.csv","candle_data\candle_data\BERA.csv", "candle_data\candle_data\ETH.csv", "candle_data\candle_data\PENGU.csv", "candle_data\candle_data\SOON.csv",
          "candle_data\candle_data\SUNDOG.csv", "candle_data\candle_data\XRP.csv", "candle_data\candle_data\YFI.csv"]:
      file_path = i
      df = pd.read_csv(file_path)

      # 데이터 확인
      df.head()

      # datetime_kst 컬럼이 없을 때만 생성
      if not df["datetime_kst"]:

            # timestamp → KST 변환
            df["datetime_kst"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Seoul")

            # 컬럼 순서 조정 (timestamp 다음에 datetime_kst 오도록)
            cols = list(df.columns)
            cols_reordered = cols[:2] + ["datetime_kst"] + cols[2:-1] + [cols[-1]]
            df = df[cols_reordered]

            # 원본 파일에 덮어쓰기 저장
            df.to_csv(file_path, index=False)




