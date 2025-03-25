import pandas as pd
from joblib import load
from etl import transformation
from pysimfin import PySimFin

def backtest_strategy(ticker, start_date, end_date, initial_capital=10000, model_path="optimized_model.joblib"):
    sf = PySimFin()
    stock_data = sf.get_share_prices(str(ticker), str(start_date), str(end_date))
    stock_data = stock_data.drop(columns=['Dividend Paid','Common Shares Outstanding'])
    stock_data.rename(columns={'Opening Price':'Open', 'Highest Price':'High','Lowest Price':'Low','Last Closing Price':'Close','Trading Volume':'Volume'},inplace=True)

    # ✅ Require minimum data before transforming (saves from crashing)
    if len(stock_data) < 90:
        return None, None, None, "Not enough raw data — select at least a 3-month range."

    # Apply feature transformation
    df_trans = transformation(stock_data)
    df_trans = df_trans.drop('Ticker',axis=1)
    debug_before_drop = len(df_trans)
    df_trans.dropna(inplace=True)
    debug_after_drop = len(df_trans)

    if len(df_trans) < 2:
        return None, None, None, f"Transformed data is too small to simulate. Started with {debug_before_drop}, kept {debug_after_drop} rows."

    model = load(model_path)

    portfolio = []
    trades = []
    cash = initial_capital
    shares = 0

    for i in range(len(df_trans) - 1):
        row = df_trans.iloc[i]
        next_row = df_trans.iloc[i + 1]

        features = row[['Low', 'Close', 'High', 'Open', 'EMA_10', 'EMA_12', 'Year']]
        prediction = model.predict([features])[0]
        next_price = next_row['Close']
        date = pd.to_datetime(next_row.name)

        action = None
        shares_traded = 0

        if prediction > row['Close'] and cash >= next_price:
            shares_traded = int(cash // next_price)
            cash -= shares_traded * next_price
            shares += shares_traded
            action = 'BUY'
        elif prediction < row['Close'] and shares > 0:
            cash += shares * next_price
            shares_traded = shares
            shares = 0
            action = 'SELL'

        if action:
            trades.append({
                "Date": date,
                "Action": action,
                "Shares": shares_traded,
                "Price": round(next_price, 2),
                "Cash": round(cash, 2)
            })

        total_value = cash + shares * next_price
        portfolio.append((date, total_value))

    if not portfolio:
        return None, None, None, "Simulation returned no portfolio data."

    df_portfolio = pd.DataFrame(portfolio, columns=["Date", "Portfolio Value"])
    df_portfolio['Date'] = pd.to_datetime(df_portfolio['Date'])
    df_portfolio = df_portfolio.set_index('Date', drop=True)
    df_trades = pd.DataFrame(trades)

    final_value = df_portfolio.iloc[-1]['Portfolio Value']
    absolute_return = final_value - initial_capital
    percent_return = (absolute_return / initial_capital) * 100

    summary = {
        "initial": initial_capital,
        "final": final_value,
        "absolute": absolute_return,
        "percent": percent_return
    }

    return df_portfolio, df_trades, summary, None
