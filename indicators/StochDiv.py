import pandas as pd
import numpy as np
from numba import njit
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal

# ==================== Swing detection (unchanged) ====================
@njit(cache=True)
def detect_swing_highs(series, left_window, confirm_bars, min_move=0.0):
    n = len(series)
    is_pivot = np.zeros(n, dtype=np.bool_)
    pivot_vals = np.zeros(n, dtype=np.float32)
    peak_idx = np.zeros(n, dtype=np.int32)

    for i in range(left_window, n - confirm_bars):
        if np.isnan(series[i]):
            continue
        left_ok = True
        min_left = np.inf
        for j in range(i - left_window, i):
            if np.isnan(series[j]):
                left_ok = False
                break
            if series[j] < min_left:
                min_left = series[j]
            if series[j] >= series[i]:
                left_ok = False
                break
        if not left_ok:
            continue
        confirm_ok = True
        for k in range(1, confirm_bars + 1):
            if np.isnan(series[i + k]):
                confirm_ok = False
                break
            if series[i + k] > series[i]:
                confirm_ok = False
                break
        if confirm_ok:
            if series[i] - min_left >= min_move:
                idx_conf = i + confirm_bars
                is_pivot[idx_conf] = True
                pivot_vals[idx_conf] = series[i]
                peak_idx[idx_conf] = i
    return is_pivot, pivot_vals, peak_idx

@njit(cache=True)
def detect_swing_lows(series, left_window, confirm_bars, min_move=0.0):
    n = len(series)
    is_pivot = np.zeros(n, dtype=np.bool_)
    pivot_vals = np.zeros(n, dtype=np.float32)
    peak_idx = np.zeros(n, dtype=np.int32)

    for i in range(left_window, n - confirm_bars):
        if np.isnan(series[i]):
            continue
        left_ok = True
        max_left = -np.inf
        for j in range(i - left_window, i):
            if np.isnan(series[j]):
                left_ok = False
                break
            if series[j] > max_left:
                max_left = series[j]
            if series[j] <= series[i]:
                left_ok = False
                break
        if not left_ok:
            continue
        confirm_ok = True
        for k in range(1, confirm_bars + 1):
            if np.isnan(series[i + k]):
                confirm_ok = False
                break
            if series[i + k] < series[i]:
                confirm_ok = False
                break
        if confirm_ok:
            if max_left - series[i] >= min_move:
                idx_conf = i + confirm_bars
                is_pivot[idx_conf] = True
                pivot_vals[idx_conf] = series[i]
                peak_idx[idx_conf] = i
    return is_pivot, pivot_vals, peak_idx

# ==================== Divergence detection adapted for Stochastic ====================
@njit(cache=True)
def bearish_divergence(high, stoch_osc,
                              price_left_window, price_confirm_bars,
                              stoch_left_window, stoch_confirm_bars,
                              lookback_bars,
                              overbought_threshold=80.0,
                              min_price_move=0.0, min_stoch_move=0.0):
    """
    Bearish divergence: price makes higher high, Stochastic makes lower high,
    while the first Stochastic high was above the overbought threshold.
    """
    price_pivot, price_vals, _ = detect_swing_highs(high, price_left_window, price_confirm_bars, min_price_move)
    stoch_pivot, stoch_vals, _ = detect_swing_highs(stoch_osc, stoch_left_window, stoch_confirm_bars, min_stoch_move)

    price_idx = np.where(price_pivot)[0]
    stoch_idx = np.where(stoch_pivot)[0]
    bearish = np.zeros(len(high), dtype=np.bool_)

    if len(price_idx) < 2 or len(stoch_idx) < 2:
        return bearish

    for i in range(1, len(price_idx)):
        curr_p_conf = price_idx[i]
        prev_ptr = i - 1
        while prev_ptr >= 0:
            prev_p_conf = price_idx[prev_ptr]
            if curr_p_conf - prev_p_conf > lookback_bars:
                break

            # find most recent Stochastic pivot <= curr_p_conf
            stoch_ptr = 0
            while stoch_ptr + 1 < len(stoch_idx) and stoch_idx[stoch_ptr + 1] <= curr_p_conf:
                stoch_ptr += 1
            if stoch_idx[stoch_ptr] > curr_p_conf:
                prev_ptr -= 1
                continue
            curr_stoch_conf = stoch_idx[stoch_ptr]

            # find Stochastic pivot <= prev_p_conf
            stoch_prev_idx = -1
            for r in range(len(stoch_idx)-1, -1, -1):
                if stoch_idx[r] <= prev_p_conf:
                    stoch_prev_idx = r
                    break
            if stoch_prev_idx == -1:
                prev_ptr -= 1
                continue
            prev_stoch_conf = stoch_idx[stoch_prev_idx]

            if (price_vals[curr_p_conf] > price_vals[prev_p_conf] and
                stoch_vals[curr_stoch_conf] < stoch_vals[prev_stoch_conf] and
                stoch_vals[prev_stoch_conf] >= overbought_threshold):
                bearish[curr_p_conf] = True
                break
            prev_ptr -= 1
    return bearish

@njit(cache=True)
def bullish_divergence(low, stoch_osc,
                              price_left_window, price_confirm_bars,
                              stoch_left_window, stoch_confirm_bars,
                              lookback_bars,
                              oversold_threshold=20.0,
                              min_price_move=0.0, min_stoch_move=0.0):
    """
    Bullish divergence: price makes lower low, Stochastic makes higher low,
    while the first Stochastic low was below the oversold threshold.
    """
    price_pivot, price_vals, _ = detect_swing_lows(low, price_left_window, price_confirm_bars, min_price_move)
    stoch_pivot, stoch_vals, _ = detect_swing_lows(stoch_osc, stoch_left_window, stoch_confirm_bars, min_stoch_move)

    price_idx = np.where(price_pivot)[0]
    stoch_idx = np.where(stoch_pivot)[0]
    bullish = np.zeros(len(low), dtype=np.bool_)

    if len(price_idx) < 2 or len(stoch_idx) < 2:
        return bullish

    for i in range(1, len(price_idx)):
        curr_p_conf = price_idx[i]
        prev_ptr = i - 1
        while prev_ptr >= 0:
            prev_p_conf = price_idx[prev_ptr]
            if curr_p_conf - prev_p_conf > lookback_bars:
                break
            stoch_ptr = 0
            while stoch_ptr + 1 < len(stoch_idx) and stoch_idx[stoch_ptr + 1] <= curr_p_conf:
                stoch_ptr += 1
            if stoch_idx[stoch_ptr] > curr_p_conf:
                prev_ptr -= 1
                continue
            curr_stoch_conf = stoch_idx[stoch_ptr]

            stoch_prev_idx = -1
            for r in range(len(stoch_idx)-1, -1, -1):
                if stoch_idx[r] <= prev_p_conf:
                    stoch_prev_idx = r
                    break
            if stoch_prev_idx == -1:
                prev_ptr -= 1
                continue
            prev_stoch_conf = stoch_idx[stoch_prev_idx]

            if (price_vals[curr_p_conf] < price_vals[prev_p_conf] and
                stoch_vals[curr_stoch_conf] > stoch_vals[prev_stoch_conf] and
                stoch_vals[prev_stoch_conf] <= oversold_threshold):
                bullish[curr_p_conf] = True
                break
            prev_ptr -= 1
    return bullish

class StochDiv:
    """
    Detects bullish/bearish divergences between price and the Stochastic oscillator.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns 'High', 'Low', 'Close'.
    stoch_k_period : int, default 14
        Lookback period for %K calculation.
    stoch_d_period : int, default 3
        Smoothing period for %D.
    stoch_slowing : int, default 3
        Internal smoothing of %K (often same as d_period).
    stoch_line : str, default 'D'
        Which Stochastic line to use for divergence detection ('K' or 'D').
    price_left_window : int, default 5
        Number of bars to the left for price swing detection.
    price_confirm_bars : int, default 2
        Number of bars to the right to confirm a price swing.
    stoch_left_window : int, default 5
        Bars to the left for Stochastic swing detection.
    stoch_confirm_bars : int, default 2
        Bars to the right to confirm a Stochastic swing.
    lookback_bars : int, default 100
        Maximum bars between two price pivots to consider a divergence.
    overbought : float, default 80.0
        Overbought threshold (Stochastic scale 0‑100).
    oversold : float, default 20.0
        Oversold threshold.
    min_price_move : float, default 0.0
        Minimum absolute price move for a swing to be valid.
    min_stoch_move : float, default 0.0
        Minimum absolute oscillator move for a swing to be valid.
    """
    def __init__(self, data,
                 stoch_k_period=14,
                 stoch_d_period=3,
                 stoch_slowing=3,
                 stoch_line='D',
                 price_left_window=3,
                 price_confirm_bars=1,
                 stoch_left_window=3,
                 stoch_confirm_bars=1,
                 lookback_bars=50,
                 overbought=80.0,
                 oversold=20.0,
                 min_price_move=0.0003,
                 min_stoch_move=2.0):
        self.data = data
        self.stoch_k_period = stoch_k_period
        self.stoch_d_period = stoch_d_period
        self.stoch_slowing = stoch_slowing
        self.stoch_line = stoch_line
        self.price_left_window = price_left_window
        self.price_confirm_bars = price_confirm_bars
        self.stoch_left_window = stoch_left_window
        self.stoch_confirm_bars = stoch_confirm_bars
        self.lookback_bars = lookback_bars
        self.overbought = overbought
        self.oversold = oversold
        self.min_price_move = min_price_move
        self.min_stoch_move = min_stoch_move

        # ---- Compute Stochastic ----
        # STOCH returns slowk, slowd
        high = data['High'].values.astype(np.float64)
        low  = data['Low'].values.astype(np.float64)
        close = data['Close'].values.astype(np.float64)

        self.slowk, self.slowd = ta.STOCH(
            high, low, close,
            fastk_period=self.stoch_k_period,
            slowk_period=self.stoch_slowing,
            slowk_matype=0,
            slowd_period=self.stoch_d_period,
            slowd_matype=0
        )
        # Choose the line to use for divergence detection
        if self.stoch_line.upper() == 'D':
            self.stoch_vals = self.slowd
        else:
            self.stoch_vals = self.slowk

        # ---- Detect divergences (reusing the njit functions) ----
        # Bearish: price highs vs oscillator highs
        self._bearish = bearish_divergence(
            high, self.stoch_vals,
            price_left_window, price_confirm_bars,
            stoch_left_window, stoch_confirm_bars,
            lookback_bars,
            overbought,
            min_price_move, min_stoch_move
        )
        # Bullish: price lows vs oscillator lows
        self._bullish = bullish_divergence(
            low, self.stoch_vals,
            price_left_window, price_confirm_bars,
            stoch_left_window, stoch_confirm_bars,
            lookback_bars,
            oversold,
            min_price_move, min_stoch_move
        )
        self.category = "divergence"

    @signal(direction="short", signal_type="discrete", weight=1.0)
    def bearish_signal(self):
        return np.where(self._bearish, -1, 0)

    @signal(direction="long", signal_type="discrete", weight=1.0)
    def bullish_signal(self):
        return np.where(self._bullish, 1, 0)

    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None: start_idx = 0
        if end_idx is None: end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]
        k_plot = pd.Series(self.slowk[start_idx:end_idx], index=df_plot.index)
        d_plot = pd.Series(self.slowd[start_idx:end_idx], index=df_plot.index)

        bearish_plot = self._bearish[start_idx:end_idx]
        bullish_plot = self._bullish[start_idx:end_idx]
        idx_bear = np.where(bearish_plot)[0]
        idx_bull = np.where(bullish_plot)[0]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.6, 0.4],
                            subplot_titles=('Price & Divergences', 'Stochastic (%K, %D)'))
        # Price
        fig.add_trace(go.Candlestick(x=df_plot.index,
                                     open=df_plot['Open'], high=df_plot['High'],
                                     low=df_plot['Low'], close=df_plot['Close'],
                                     name='Price'), row=1, col=1)
        # Bearish signals
        fig.add_trace(go.Scatter(x=df_plot.index[idx_bear],
                                 y=df_plot['High'].iloc[idx_bear] * 1.001,
                                 mode='markers',
                                 marker=dict(color='red', size=12, symbol='arrow-down'),
                                 name='Bearish Div'), row=1, col=1)
        # Bullish signals
        fig.add_trace(go.Scatter(x=df_plot.index[idx_bull],
                                 y=df_plot['Low'].iloc[idx_bull] * 0.999,
                                 mode='markers',
                                 marker=dict(color='lime', size=12, symbol='arrow-up'),
                                 name='Bullish Div'), row=1, col=1)
        # Stochastic lines
        fig.add_trace(go.Scatter(x=k_plot.index, y=k_plot,
                                 mode='lines', line=dict(color='blue', width=1.5),
                                 name='%K'), row=2, col=1)
        fig.add_trace(go.Scatter(x=d_plot.index, y=d_plot,
                                 mode='lines', line=dict(color='orange', width=2),
                                 name='%D'), row=2, col=1)
        # Overbought / oversold
        fig.add_hline(y=self.overbought, line_dash="dash", line_color="red",
                      annotation_text="Overbought", row=2, col=1)
        fig.add_hline(y=self.oversold, line_dash="dash", line_color="green",
                      annotation_text="Oversold", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray",
                      annotation_text="50", row=2, col=1)

        fig.update_layout(title='Stochastic Divergence',
                          xaxis_title='Date', yaxis_title='Price',
                          height=800, width=1000, template='plotly_dark',
                          hovermode='x unified',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        fig.update_yaxes(title_text="Stochastic", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()