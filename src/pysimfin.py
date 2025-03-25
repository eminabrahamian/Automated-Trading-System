import requests
import pandas as pd
import os
import logging
import re
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PySimFin():
    def __init__(self):
        #self.api_key = os.getenv('API_KEY') # for running locally
        self.api_key = st.secrets["SIMFIN_API_KEY"]
        self.url = "https://backend.simfin.com/api/v3/companies"
        self.headers = {"accept": "application/json", "Authorization": f"Bearer {self.api_key}"}

    def get_available_tickers(self):
        unique_tickers = pd.read_csv(f'./data/raw/us-companies.csv', sep=';', index_col=0).index.unique().to_list()[1:]
        return [item for item in unique_tickers if re.fullmatch(r'[A-Za-z0-9]+', item)]

    def get_available_dates(self, ticker):
        url_endpoint = self.url + f'/prices/compact?&ticker={ticker}'
        response = requests.get(url_endpoint, headers=self.headers)
        df = pd.DataFrame(response.json()[0]['data'], columns=response.json()[0]['columns'])
        return pd.date_range(start=df['Date'].min(), end=df['Date'].max()).strftime('%Y-%m-%d').tolist()

    def validate_inputs(self, ticker, start, end):
        """Validates the ticker and date inputs."""
        if not isinstance(ticker, str) or not isinstance(start, str) or not isinstance(end, str):
            #logging.error("Ticker, start, and end must be strings.")
            raise TypeError("Ticker, start, and end must be strings.")

        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(date_pattern, start) or not re.match(date_pattern, end):
            #logging.error("Start and end dates must be in format YYYY-MM-DD.")
            raise ValueError("Start and end dates must be in format YYYY-MM-DD.")

        # Extract parts
        year_s, month_s, day_s = start.split("-")
        year_e, month_e, day_e = end.split("-")

        # Check valid month (1-12) and day (1-31)
        if not (1 <= int(month_s) <= 12):
            raise ValueError(f"Month in '{start}' is out of range (1-12).")
        if not (1 <= int(month_e) <= 12):
            raise ValueError(f"Month in '{end}' is out of range (1-12).")
        if not (1 <= int(day_s) <= 31):
            raise ValueError(f"Day in '{start}' is out of range (1-31).")
        if not (1 <= int(day_e) <= 31):
            raise ValueError(f"Day in '{end}' is out of range (1-31).")

        # Fetch available date range from API
        available_dates = self.get_available_dates(ticker)
        if start not in available_dates or end not in available_dates:
            logging.error("Date range is outside of available SimFin data.")
            raise IndexError("Date range is outside of available SimFin data.")

    def get_share_prices(self, ticker, start, end):
        """Fetches share prices for a given ticker and date range."""
        self.validate_inputs(ticker, start, end)

        url_endpoint = self.url + f'/prices/compact?ticker={ticker}&start={start}&end={end}'
        response = requests.get(url_endpoint, headers=self.headers)

        if response.status_code != 200 or not response.json():
            #logging.warning(f"No data available for ticker {ticker} in the given range {start} to {end}.")
            raise ValueError(f"No available data found for ticker {ticker} in the given range {start} to {end}.")

        try:
            df = pd.DataFrame(response.json()[0]['data'], columns=response.json()[0]['columns'])
            df['Ticker'] = ticker
            df['SimFinId'] = response.json()[0]['id']
            cols = ['Ticker', 'SimFinId', 'Date', 'Opening Price', 'Highest Price', 'Lowest Price',
                    'Last Closing Price', 'Adjusted Closing Price', 'Trading Volume', 'Dividend Paid',
                    'Common Shares Outstanding']
            df = df[cols]
            df = df.rename(columns={'Last Closing Price':'Close'})
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date', drop=True)
            return df
        except Exception as e:
            #logging.error(f"Error processing share prices: {str(e)}")
            raise RuntimeError(f"Error processing share prices data. \n{e}")

    def get_financial_statement(self, ticker, start, end, statements):
        """Fetches financial statements for a given ticker and date range."""
        self.validate_inputs(ticker, start, end)

        if not isinstance(statements, list):
            #logging.error("Statements parameter must be a list.")
            raise TypeError("Statements parameter must be a list.")

        valid_statements = ['PL', 'BS', 'CF', 'DERIVED']
        if not all(stmt in valid_statements for stmt in statements):
            #logging.error("Invalid statement type found in list.")
            raise ValueError("One or more statements are invalid. Allowed: ['PL', 'BS', 'CF', 'DERIVED']")

        url_endpoint = self.url + f'/statements/compact?ticker={ticker}&start={start}&end={end}&statements={",".join(statements)}'
        response = requests.get(url_endpoint, headers=self.headers)

        if response.status_code != 200 or not response.json():
            #logging.warning(
            #    f"No financial statement data available for ticker {ticker} in the given range {start} to {end}.")
            raise ValueError(f"No available data found for ticker {ticker} in the given range {start} to {end}.")

        try:
            dfs = {}
            for idx, item in enumerate(statements):
                info = response.json()[0]['statements'][idx]
                dfs[f"df_{item}"] = pd.DataFrame(info['data'], columns=info['columns'])
                dfs[f"df_{item}"]['Fiscal Year'] = dfs[f"df_{item}"]['Fiscal Year'].astype(int)
            return dfs
        except Exception as e:
            #logging.error(f"Error processing financial statements: {str(e)}")
            raise RuntimeError("Error processing financial statements data.")