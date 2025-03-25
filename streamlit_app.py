import streamlit as st
import datetime
import math
import pandas as pd
from pysimfin import PySimFin
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from joblib import load
from etl import transformation
import logging

# NEW: Import backtesting function
from backtesting import backtest_strategy

# Configure logging
logging.basicConfig(level=logging.ERROR)

def remove_empty_columns(df):
    return df.dropna(axis=1, how='all') if df is not None and not df.empty else df

# Initialize PySimFin object
sf = PySimFin()

# Streamlit config
st.set_page_config(page_title="Automated Trading System", layout="wide")
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("➡️ Go to", ["Home", "Go Live", "Trading Advice"])

# ===================== HOME PAGE =====================
if page == "Home":
    st.title("📊 Automated Trading System")
    st.markdown("""
    Welcome to our Automated Stock Trading Dashboard. This platform combines machine learning-powered price forecasting with an interactive, real-time trading assistant. Users can explore historical and current market data for selected U.S. companies, view next-day price predictions, and receive actionable buy/sell/hold signals based on our predictive models.
    \n\nCore features include:
    \n- Real-time stock price data retrieval via the SimFin API
    \n- Next-day stock price prediction using a trained ML model
    \n- Dynamic trading suggestions tailored to a custom investment budget
    \n- Profit/loss estimation for informed trading decisions
    \n- Interactive data visualizations for trend exploration and analysis
    """)

    st.subheader("👨‍💻 Meet the Team")
    st.markdown("""
    - EMIN ABRAHAMIAN  
    - IGNACIO RODRÍGUEZ  
    - MIGUEL RODRÍGUEZ  
    - NICOLE BATINOVICH  
    - ABDULRAHMAN ALABDULKARIM  
    """)

    st.subheader("🎯 Purpose")
    st.markdown("""
    The goal of this system is to support individual traders and financial enthusiasts in making data-driven daily trading decisions. By combining historical data analysis, financial modeling, and a responsive web interface, our platform empowers users to:
    \n- Forecast short-term stock movements
    \n- Simulate trading strategies with custom budgets
    \n- Understand potential risks and returns before acting
    \n- Interact with a simplified yet powerful trading assistant
    \n\n\nWhether you're new to trading or testing investment strategies, this tool offers an educational and practical environment to refine your approach.
    """)

# ===================== GO LIVE PAGE =====================
elif page == "Go Live":
    st.title("🚀 Go Live - Real-Time Stock Analysis")

    # Sidebar Filter Selection
    st.sidebar.header("📌 Filters")

    selected_ticker = st.sidebar.selectbox(
    "Choose a Company",
    options=sf.get_available_tickers())
    start_dt = pd.to_datetime(sf.get_available_dates(str(selected_ticker))[0])
    end_dt = pd.to_datetime(sf.get_available_dates(str(selected_ticker))[-1])
    start_date = st.sidebar.date_input("📅 Start Date", start_dt)
    end_date = st.sidebar.date_input("📅 End Date", end_dt)
    stock_data = sf.get_share_prices(str(selected_ticker), str(start_date), str(end_date))
    statements = st.sidebar.multiselect("📌 Select Financial Statements", options=['PL', 'BS', 'CF', 'DERIVED'])

    # ========== PRICE CHART ==========
    # ✅ Live Stock Price Card

    st.header('Interactive Stock Data')

    if stock_data is not None and len(stock_data) > 1:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(stock_data.index, stock_data['Close'], label="Closing Price", color="#1f77b4", linewidth=2)
        ax.set_title(f"'{selected_ticker}' Closing Price Trend", fontsize=16)

        # Format x-axis to show major ticks every month
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show a tick every month
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format like 'Mar 2025'
        plt.setp(ax.get_xticklabels(), rotation=45)

        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Not enough data to show chart.")

    # ============ FINANCIAL STATEMENTS ============
    st.header('Interactive Financial Statements')
    financial_data = sf.get_financial_statement(str(selected_ticker), str(start_date), str(end_date), statements)

    for item in statements:
        st.subheader(item)
        st.dataframe(financial_data[f'df_{item}'],
                     column_config={'Fiscal Year':st.column_config.NumberColumn(format="%d")})

# ===================== TRADING ADVICE PAGE =====================
elif page == "Trading Advice":
    st.title("🚀 Trading Advice - Data-Driven Investment Strategy")

    # Sidebar Filter Selection
    st.sidebar.header("📌 Filters")

    selected_ticker = st.sidebar.selectbox(
        "Choose a Company",
        options=sf.get_available_tickers())
    start_dt = pd.to_datetime(sf.get_available_dates(str(selected_ticker))[0])
    end_dt = pd.to_datetime(sf.get_available_dates(str(selected_ticker))[-1])
    start_date = st.sidebar.date_input("📅 Start Date", start_dt)
    end_date = st.sidebar.date_input("📅 End Date", end_dt)
    stock_data = sf.get_share_prices(str(selected_ticker), str(start_date), str(end_date))

    st.header("Next Day Closing Price Prediction")

    budget = st.number_input("Enter your trading budget (USD):", min_value=1.0)

    try:
        model = load("optimized_model.joblib")

        if stock_data is not None and not stock_data.empty:
            df_pred = stock_data.copy()
            df_pred = df_pred.drop(columns=['Dividend Paid', 'Common Shares Outstanding'])
            df_pred.rename(columns={'Opening Price': 'Open', 'Highest Price': 'High', 'Lowest Price': 'Low',
                                       'Last Closing Price': 'Close', 'Trading Volume': 'Volume'}, inplace=True)

            df_transformed = transformation(df_pred)
            df_transformed = df_transformed.drop('Ticker', axis=1)
            df_transformed = remove_empty_columns(df_transformed).dropna()

            if not df_transformed.empty:
                latest_row = df_transformed.tail(1)
                input_features = latest_row[
                    ['Low', 'Close', 'High', 'Open', 'EMA_10', 'EMA_12', 'Year']]
                predicted_close = model.predict(input_features)[0]
                last_close = latest_row['Close'].values[0]

                st.metric(label="📉 Last Close Price", value=f"${last_close:.2f}")
                st.metric(label="📈 Predicted Next-Day Close", value=f"${predicted_close:.2f}")

                def round_down_to_quarter(value):
                    return math.floor(value * 4) / 4

                if budget > 0 and last_close > 0:
                    raw_shares = budget / last_close
                    shares = round_down_to_quarter(raw_shares)  # Allow buying in 0.25 increments

                    if predicted_close > last_close:
                        profit = (predicted_close - last_close) * shares
                        st.success("📢 **Signal: BUY**")
                        st.write(f"You can buy **{shares:.2f} shares** at ${last_close:.2f} each.")
                        st.write(f"📈 Predicted profit: **${profit:.2f}**")

                    elif predicted_close < last_close:
                        loss_avoided = (last_close - predicted_close) * shares
                        st.error("📢 **Signal: SELL**")
                        st.write(f"You can sell **{shares:.2f} shares** at ${last_close:.2f} each.")
                        st.write(f"📉 Predicted losses avoided: **${loss_avoided:.2f}**")

                    else:
                        st.info("📢 **Signal: HOLD**")
                        st.write("The predicted price is the same as the current price. No action suggested.")

                else:
                    st.warning("⚠️ Please enter a valid budget and price.")
            else:
                st.warning("⚠️ Not enough data to make a prediction.")

    except Exception as e:
        logging.error(f"❌ Error loading model or making prediction: {e}")
        st.error(f"❌ Error loading model or making prediction: {e}")

    # ============ 📊 BACKTESTING SIMULATOR ============
    st.header("📊 Backtesting Simulator")
    st.markdown("Test a historical investment strategy using our model.")
    st.markdown("#### 📌 Select Backtest Options")

    backtest_ticker = st.selectbox(
        "Select a Company for Backtesting",
        options=sf.get_available_tickers())

    backtest_start = st.date_input("🕰️ Backtest Start Date", datetime.date(2020, 1, 1))
    initial_capital = st.number_input("💰 Initial Investment ($)", value=10000, min_value=1000, step=500)

    if st.button("Run Backtest"):
        backtest_df, trade_log, summary, error_msg = backtest_strategy(
            ticker=backtest_ticker,
            start_date=backtest_start,
            end_date=end_date,
            initial_capital=initial_capital
        )

        if error_msg:
            st.error(f"⚠️ {error_msg}")
            st.info("Try a different start date or ticker with more data.")
        else:
            st.subheader("📈 Portfolio Value Over Time")
            st.line_chart(backtest_df)

            st.subheader("📋 Trade Log")
            if trade_log is not None and not trade_log.empty:
                st.dataframe(trade_log)
            else:
                st.info("No trades were executed during this backtest.")

            st.subheader("📊 Return Summary")
            st.metric("💰 Initial Investment", f"${summary['initial']:,.2f}")
            st.metric("📈 Final Portfolio Value", f"${summary['final']:,.2f}")
            st.metric("📈 Absolute Return", f"${summary['absolute']:,.2f}")
            st.metric("📈 Percentage Return", f"{summary['percent']:.2f}%")

# Footer
st.markdown("---")
st.markdown("🚀 Built with SimFin API & Streamlit | Group 7 - Automated Trading System")
