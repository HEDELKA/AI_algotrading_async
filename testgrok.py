import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from binance.client import Client  # Requires binance library: pip install python-binance
import time
import itertools

def fetch_klines(symbol='BTCUSDT', interval='5m', total_bars=1200, client=None):
    if client is None:
        client = Client()
    limit = 1000
    data = []
    end_time = None

    while len(data) < total_bars:
        bars_to_fetch = min(limit, total_bars - len(data))
        try:
            klines = client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=bars_to_fetch,
                endTime=end_time
            )
        except Exception as e:
            print("Binance API Error:", e)
            break
        if not klines:
            break
        data = klines + data
        end_time = klines[0][0] - 1
        time.sleep(0.1)
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    df = df.drop_duplicates('timestamp')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def calculate_indicators(df, ema_period=50, macd_fast=12, macd_slow=26, macd_signal=9, bb_period=20, bb_std=2.0, atr_period=14, vol_period=20):
    # EMA
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()

    # MACD
    ema_fast = df['close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=macd_slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(bb_period).mean()
    df['bb_std'] = df['close'].rolling(bb_period).std()
    df['bb_upper'] = df['bb_mid'] + bb_std * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - bb_std * df['bb_std']

    # ATR
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(atr_period).mean()

    # Volume average
    df['vol_avg'] = df['volume'].rolling(vol_period).mean()

    # Bollinger width %
    df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_mid']) * 100

    return df

def detect_qml_bull(df, i, order=5, recency=30):
    """
    Detect bullish Quasimodo: sweep below prior low, then higher low.
    Returns the sweep low (extreme) if detected, else None.
    """
    prices = df['low'][:i+1].values
    local_min_idx = argrelextrema(prices, np.less, order=order)[0]
    if len(local_min_idx) < 3:
        return None
    idx3 = local_min_idx[-1]
    idx2 = local_min_idx[-2]
    idx1 = local_min_idx[-3]
    low1 = prices[idx1]
    low2 = prices[idx2]
    low3 = prices[idx3]
    if low2 < low1 and low3 > low2 and (i - idx3 < recency):  # Recent formation
        return low2
    return None

def detect_qml_bear(df, i, order=5, recency=30):
    """
    Detect bearish Quasimodo: sweep above prior high, then lower high.
    Returns the sweep high (extreme) if detected, else None.
    """
    prices = df['high'][:i+1].values
    local_max_idx = argrelextrema(prices, np.greater, order=order)[0]
    if len(local_max_idx) < 3:
        return None
    idx3 = local_max_idx[-1]
    idx2 = local_max_idx[-2]
    idx1 = local_max_idx[-3]
    high1 = prices[idx1]
    high2 = prices[idx2]
    high3 = prices[idx3]
    if high2 > high1 and high3 < high2 and (i - idx3 < recency):  # Recent formation
        return high2
    return None

def backtest_strategy(df, use_vol_filter=True, squeeze_threshold=1.5, atr_sl_mult=1.0, rr=2.0, qml_order=5, qml_recency=30, use_qml_extreme_sl=True):
    in_position = False
    trades = []
    position = None

    max_lookback = 100  # For indicator validity
    for i in range(max_lookback, len(df)):
        if not in_position:
            # Filters (weakened: optional vol, lower squeeze)
            vol_filter_pass = (not use_vol_filter) or (df['volume'][i] > df['vol_avg'][i] * 0.8)  # Lowered to 80% of avg
            no_squeeze = df['bb_width'][i] >= squeeze_threshold
            if not (vol_filter_pass and no_squeeze):
                continue

            # Long entry (simplified: removed close above mid as mandatory, but use for confirmation)
            macd_cross_up = (df['macd'][i] > df['signal'][i]) and (df['macd'][i-1] <= df['signal'][i-1])
            price_above_ema = df['close'][i] > df['ema'][i]
            qml_low = detect_qml_bull(df, i, order=qml_order, recency=qml_recency)
            if macd_cross_up and price_above_ema and qml_low is not None:
                entry = df['close'][i]
                if use_qml_extreme_sl:
                    sl = qml_low - atr_sl_mult * df['atr'][i]
                else:
                    sl = entry - atr_sl_mult * df['atr'][i]
                risk = entry - sl
                tp = entry + rr * risk
                position = {
                    'entry_idx': i,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'type': 'long',
                    'reached_1to1': False
                }
                in_position = True
                continue

            # Short entry
            macd_cross_down = (df['macd'][i] < df['signal'][i]) and (df['macd'][i-1] >= df['signal'][i-1])
            price_below_ema = df['close'][i] < df['ema'][i]
            qml_high = detect_qml_bear(df, i, order=qml_order, recency=qml_recency)
            if macd_cross_down and price_below_ema and qml_high is not None:
                entry = df['close'][i]
                if use_qml_extreme_sl:
                    sl = qml_high + atr_sl_mult * df['atr'][i]
                else:
                    sl = entry + atr_sl_mult * df['atr'][i]
                risk = sl - entry
                tp = entry - rr * risk
                position = {
                    'entry_idx': i,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'type': 'short',
                    'reached_1to1': False
                }
                in_position = True
                continue

        else:
            # Check exits and trailing (simplified: trail earlier if desired)
            if position['type'] == 'long':
                risk = position['entry'] - position['sl']
                if not position['reached_1to1'] and df['high'][i] >= position['entry'] + risk:
                    position['reached_1to1'] = True
                if position['reached_1to1']:
                    position['sl'] = max(position['sl'], df['bb_mid'][i])

                hit_sl = df['low'][i] <= position['sl']
                hit_tp = df['high'][i] >= position['tp']
                macd_cross_down = (df['macd'][i] < df['signal'][i]) and (df['macd'][i-1] >= df['signal'][i-1])
                hit_upper = df['high'][i] >= df['bb_upper'][i]

                if hit_sl or hit_tp or macd_cross_down or hit_upper:
                    if hit_sl:
                        exit_price = position['sl']
                    elif hit_tp:
                        exit_price = position['tp']
                    elif hit_upper:
                        exit_price = df['bb_upper'][i]
                    else:
                        exit_price = df['close'][i]
                    pnl = (exit_price - position['entry']) / position['entry'] * 100  # % return
                    trades.append({
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'entry_price': position['entry'],
                        'exit_price': exit_price,
                        'pnl_pct': pnl,
                        'type': 'long'
                    })
                    in_position = False
                    position = None

            else:  # short
                risk = position['sl'] - position['entry']
                if not position['reached_1to1'] and df['low'][i] <= position['entry'] - risk:
                    position['reached_1to1'] = True
                if position['reached_1to1']:
                    position['sl'] = min(position['sl'], df['bb_mid'][i])

                hit_sl = df['high'][i] >= position['sl']
                hit_tp = df['low'][i] <= position['tp']
                macd_cross_up = (df['macd'][i] > df['signal'][i]) and (df['macd'][i-1] <= df['signal'][i-1])
                hit_lower = df['low'][i] <= df['bb_lower'][i]

                if hit_sl or hit_tp or macd_cross_up or hit_lower:
                    if hit_sl:
                        exit_price = position['sl']
                    elif hit_tp:
                        exit_price = position['tp']
                    elif hit_lower:
                        exit_price = df['bb_lower'][i]
                    else:
                        exit_price = df['close'][i]
                    pnl = (position['entry'] - exit_price) / position['entry'] * 100  # % return
                    trades.append({
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'entry_price': position['entry'],
                        'exit_price': exit_price,
                        'pnl_pct': pnl,
                        'type': 'short'
                    })
                    in_position = False
                    position = None

    if trades:
        trades_df = pd.DataFrame(trades)
        total_pnl_pct = trades_df['pnl_pct'].sum()
        win_rate = (trades_df['pnl_pct'] > 0).mean() * 100
        num_trades = len(trades_df)
        avg_pnl = trades_df['pnl_pct'].mean()
        return {
            'total_pnl_pct': total_pnl_pct,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'avg_pnl': avg_pnl,
            'trades_df': trades_df
        }
    else:
        return {
            'total_pnl_pct': 0,
            'win_rate': 0,
            'num_trades': 0,
            'avg_pnl': 0,
            'trades_df': pd.DataFrame()
        }
import matplotlib.pyplot as plt

def plot_equity_curve(df, trades_df):
    """
    Строит equity curve (кривую капитала) по сделкам.
    df — основной датафрейм свечей
    trades_df — датафрейм сделок из backtest_strategy
    """
    if trades_df.empty:
        print("Нет сделок для построения equity curve.")
        return

    # Создаём массив equity (рост депо)
    equity = [1.0]  # старт = 1.0 (100%)
    timestamps = [df['timestamp'].iloc[trades_df['entry_idx'].iloc[0]]]  # первая сделка

    current_equity = 1.0
    for _, trade in trades_df.iterrows():
        pnl_mult = 1 + trade['pnl_pct'] / 100
        current_equity *= pnl_mult

        equity.append(current_equity)
        timestamps.append(df['timestamp'].iloc[int(trade['exit_idx'])])

    # График
    plt.figure(figsize=(14, 6))
    plt.plot(timestamps, equity, linewidth=2)
    plt.title("Equity Curve")
    plt.xlabel("Дата")
    plt.ylabel("Капитал (x)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
best_df = None
def grid_search(symbol='BTCUSDT', total_bars=5000):
    param_grid = {
        'interval': ['1h'],
        'ema_period': [100],
        'bb_std': [2.0],
        'atr_sl_mult': [1.5],
        'rr': [2],
        'qml_order': [3],
        'qml_recency': [20],
        'use_vol_filter': [False],
        'squeeze_threshold': [1.0],
        'use_qml_extreme_sl': [False]
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))

    best_result = None
    best_params = None
    best_score = float('-inf')  # Maximize total PNL, but require min trades

    for combo in combos:
        params = dict(zip(keys, combo))
        print(f"Testing params: {params}")

        df = fetch_klines(symbol=symbol, interval=params['interval'], total_bars=total_bars)
        df = calculate_indicators(df, ema_period=params['ema_period'], bb_std=params['bb_std'])
        result = backtest_strategy(df, use_vol_filter=params['use_vol_filter'], squeeze_threshold=params['squeeze_threshold'],
                                  atr_sl_mult=params['atr_sl_mult'], rr=params['rr'], qml_order=params['qml_order'],
                                  qml_recency=params['qml_recency'], use_qml_extreme_sl=params['use_qml_extreme_sl'])

        # Score: total_pnl if num_trades >= 10, else penalize
        score = result['total_pnl_pct'] if result['num_trades'] >= 10 else result['total_pnl_pct'] - 1000  # Heavy penalty for few trades

        if score > best_score:
            best_score = score
            best_params = params
            best_result = result
            best_df = df.copy()
    if best_result:
        print("\nBest Parameters:")
        print(best_params)
        print(f"Best Backtest Results:\nTotal PNL (%): {best_result['total_pnl_pct']:.2f}%\nWin Rate: {best_result['win_rate']:.2f}%\nNumber of Trades: {best_result['num_trades']}\nAverage PNL: {best_result['avg_pnl']:.2f}%")

        # --- NEW: построение equity curve ---
        plot_equity_curve(best_df, best_result['trades_df'])
    else:
        print("No valid results found.")

# Example usage
if __name__ == "__main__":

    grid_search(symbol='LTCUSDT', total_bars=5000)  # This will take time due to many combos; reduce grid for testing
