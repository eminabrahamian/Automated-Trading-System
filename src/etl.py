import numpy as np
import pandas as pd
import zipfile
import os

def transformation(df):
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['DayOfWeek'] = df.index.day_of_week
    df['Quarter'] = df.index.quarter
    df['DayOfYear'] = df.index.dayofyear
    df['WeekOfYear'] = df.index.isocalendar().week
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    def compute_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['RSI'] = compute_rsi(df['Close'])
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['Rolling_Std_10'] = df['Daily_Return'].rolling(window=10).std()
    df['Rolling_Std_20'] = df['Daily_Return'].rolling(window=20).std()
    df['Rolling_Std_50'] = df['Daily_Return'].rolling(window=50).std()

    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))

    df['TrueRange'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()

    df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()

    return df

def load_and_prepare_data():
    companies_df = pd.read_csv('../data/raw/us-companies.csv', sep=';', index_col=0)

    zip_path = "../data/raw/us-shareprices-daily.csv.zip"
    extract_to = "../data/raw/"

    # Unzip the file if not already extracted
    csv_file = os.path.join(extract_to, "us-shareprices-daily.csv")
    if not os.path.exists(csv_file):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    shareprices_df = pd.read_csv('../data/raw/us-shareprices-daily.csv', sep=';', parse_dates=['Date'], index_col=0)
    shareprices_df = shareprices_df[shareprices_df.index=='TSLA']
    shareprices_df = shareprices_df.set_index('Date', drop=True)
    companies_df = companies_df[~companies_df.index.isna()]
    print(companies_df.head())

    shareprices_df = shareprices_df.drop(columns=['Dividend', 'Shares Outstanding', 'SimFinId'])
    
    df = transformation(shareprices_df)
    df['next_day_close'] = df['Close'].shift(-1)
     
    cols = list(df.columns)
    def values_imputation(df,cols):
        median_values = df[cols].median()
        filled = df[cols].fillna(median_values)
        return filled
    df_na_filled = values_imputation(df,cols)
    
    return df_na_filled

if __name__ == '__main__':
    df_final = load_and_prepare_data()
    df_final.to_csv('../data/processed/final_etl_df.csv', index=True)
    print("Data saved to 'final_etl_df.csv'.")