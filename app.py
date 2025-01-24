import streamlit as st
import pandas as pd
import yfinance as yf
import ccxt
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def fetch_crypto_data(symbol, timeframe, start_date):
    exchange = ccxt.binance()
    since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def fetch_stock_data(symbol, start_date):
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date)
    return df

def detect_internal_orderblocks(df, atr_period=14, atr_multiplier=1.5):
    # Calculate ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(atr_period).mean()
    
    # Initialize arrays for order blocks
    bullish_ob = np.zeros(len(df))
    bearish_ob = np.zeros(len(df))
    
    # Detect internal order blocks
    for i in range(3, len(df)):
        # Bullish OB
        if (df['low'].iloc[i-1] < df['low'].iloc[i-2] and 
            df['high'].iloc[i] > df['high'].iloc[i-1]):
            if (df['high'].iloc[i-2] - df['low'].iloc[i-2]) < atr.iloc[i-2] * atr_multiplier:
                bullish_ob[i-2] = 1
                
        # Bearish OB
        if (df['high'].iloc[i-1] > df['high'].iloc[i-2] and 
            df['low'].iloc[i] < df['low'].iloc[i-1]):
            if (df['high'].iloc[i-2] - df['low'].iloc[i-2]) < atr.iloc[i-2] * atr_multiplier:
                bearish_ob[i-2] = 1
    
    return bullish_ob, bearish_ob

def backtest_strategy(df, initial_capital=10000):
    bullish_ob, bearish_ob = detect_internal_orderblocks(df)
    
    position = 0  # 0: no position, 1: long position
    capital = initial_capital
    trades = []
    equity_curve = [initial_capital]
    
    for i in range(1, len(df)):
        if bullish_ob[i-1] and position == 0:
            # Buy signal
            position = 1
            entry_price = df['close'].iloc[i]
            trades.append({
                'type': 'buy',
                'entry_date': df.index[i],
                'entry_price': entry_price,
                'size': capital / entry_price
            })
        elif bearish_ob[i-1] and position == 1:
            # Sell signal
            position = 0
            exit_price = df['close'].iloc[i]
            trades[-1].update({
                'exit_date': df.index[i],
                'exit_price': exit_price,
                'pnl': (exit_price - trades[-1]['entry_price']) * trades[-1]['size']
            })
            trades[-1]['return_pct'] = ((exit_price - trades[-1]['entry_price']) / trades[-1]['entry_price']) * 100
            capital += trades[-1]['pnl']
        
        # Update equity curve
        if position == 0:
            equity_curve.append(capital)
        else:
            current_value = capital + (df['close'].iloc[i] - trades[-1]['entry_price']) * trades[-1]['size']
            equity_curve.append(current_value)
    
    return pd.DataFrame(trades), pd.Series(equity_curve, index=df.index)

def calculate_metrics(trades_df, equity_curve):
    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        
    # Calculate basic metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_profit = trades_df['pnl'].mean()
    
    # Calculate max drawdown
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()
    
    # Calculate Sharpe ratio
    returns = equity_curve.pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate * 100,
        'avg_profit': avg_profit,
        'max_drawdown': max_drawdown * 100,
        'sharpe_ratio': sharpe_ratio
    }

def main():
    st.title("Smart Money Concepts Backtester")
    
    # Sidebar inputs
    st.sidebar.header("Settings")
    asset_type = st.sidebar.selectbox("Asset Type", ["Crypto", "Stock"])
    symbol = st.sidebar.text_input("Symbol", "BTC/USDT" if asset_type == "Crypto" else "AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    timeframe = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d"]) if asset_type == "Crypto" else "1d"
    
    if st.sidebar.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            # Fetch data
            if asset_type == "Crypto":
                df = fetch_crypto_data(symbol, timeframe, start_date.strftime('%Y-%m-%d'))
            else:
                df = fetch_stock_data(symbol, start_date)
            
            # Run backtest
            trades_df, equity_curve = backtest_strategy(df)
            metrics = calculate_metrics(trades_df, equity_curve)
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Trades", f"{metrics['total_trades']}")
            col2.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
            col3.metric("Avg Profit", f"${metrics['avg_profit']:.2f}")
            col4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
            col5.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            
            # Create plot
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Price chart with signals
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price"
            ), row=1, col=1)
            
            # Add buy/sell signals
            for _, trade in trades_df.iterrows():
                fig.add_scatter(x=[trade['entry_date']], y=[trade['entry_price']],
                              mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
                              name='Buy', row=1, col=1)
                if 'exit_price' in trade:
                    fig.add_scatter(x=[trade['exit_date']], y=[trade['exit_price']],
                                  mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
                                  name='Sell', row=1, col=1)
            
            # Equity curve
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                name="Equity Curve",
                line=dict(color='blue')
            ), row=2, col=1)
            
            fig.update_layout(height=800, title=f"{symbol} Backtest Results")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display trades table
            if len(trades_df) > 0:
                st.subheader("Trade History")
                trades_display = trades_df.copy()
                trades_display['return_pct'] = trades_display['return_pct'].round(2).astype(str) + '%'
                st.dataframe(trades_display)

if __name__ == "__main__":
    main()