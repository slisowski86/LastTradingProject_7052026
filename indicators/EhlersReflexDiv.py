import pandas as pd
import numpy as np
from numba import njit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal  # your decorator
import talib as ta

# =============================================================================
# 1. Numba‑accelerated Ehlers Reflex core
# =============================================================================
@njit(cache=True)
def ehlers_reflex_core(close, reflex_period=20, reflex_alpha=0.8):
    """
    Compute the Ehlers Reflex oscillator.
    
    Reflex = HP(close) / MAD(HP, period)
    where HP is a high‑pass filter:
        HP[i] = 0.5 * (1+alpha) * (close[i] - close[i-1]) + alpha * HP[i-1]
    """
    n = len(close)
    hp = np.zeros(n)
    for i in range(1, n):
        hp[i] = 0.5 * (1.0 + reflex_alpha) * (close[i] - close[i-1]) + reflex_alpha * hp[i-1]

    # Mean absolute deviation of HP over reflex_period
    mad = np.full(n, np.nan)
    for i in range(reflex_period, n):
        window = hp[i - reflex_period + 1 : i + 1]
        mad[i] = np.mean(np.abs(window))

    reflex = np.full(n, np.nan)
    for i in range(reflex_period, n):
        if mad[i] > 1e-10:
            reflex[i] = hp[i] / mad[i]
    return reflex

# =============================================================================
# 2. Generic swing detection (unchanged, works for any array)
# =============================================================================
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

# =============================================================================
# 3. Generic divergence detection (oscillator‑agnostic parameter names)
# =============================================================================
@njit(cache=True)
def bearish_divergence(high, osc,                     # osc = oscillator array
                       price_left_window, price_confirm_bars,
                       osc_left_window, osc_confirm_bars,
                       lookback_bars,
                       overbought_threshold=70.0,
                       min_price_move=0.0, min_osc_move=0.0):
    """
    Bearish divergence: price makes higher high, oscillator makes lower high,
    first oscillator high above overbought_threshold.
    """
    price_pivot, price_vals, _ = detect_swing_highs(high, price_left_window, price_confirm_bars, min_price_move)
    osc_pivot, osc_vals, _ = detect_swing_highs(osc, osc_left_window, osc_confirm_bars, min_osc_move)

    price_idx = np.where(price_pivot)[0]
    osc_idx = np.where(osc_pivot)[0]
    bearish = np.zeros(len(high), dtype=np.bool_)

    if len(price_idx) < 2 or len(osc_idx) < 2:
        return bearish

    for i in range(1, len(price_idx)):
        curr_p_conf = price_idx[i]
        prev_ptr = i - 1
        while prev_ptr >= 0:
            prev_p_conf = price_idx[prev_ptr]
            if curr_p_conf - prev_p_conf > lookback_bars:
                break

            # most recent oscillator pivot <= current price pivot
            osc_ptr = 0
            while osc_ptr + 1 < len(osc_idx) and osc_idx[osc_ptr + 1] <= curr_p_conf:
                osc_ptr += 1
            if osc_idx[osc_ptr] > curr_p_conf:
                prev_ptr -= 1
                continue
            curr_osc_conf = osc_idx[osc_ptr]

            # oscillator pivot <= previous price pivot
            osc_prev_idx = -1
            for r in range(len(osc_idx)-1, -1, -1):
                if osc_idx[r] <= prev_p_conf:
                    osc_prev_idx = r
                    break
            if osc_prev_idx == -1:
                prev_ptr -= 1
                continue
            prev_osc_conf = osc_idx[osc_prev_idx]

            if (price_vals[curr_p_conf] > price_vals[prev_p_conf] and
                osc_vals[curr_osc_conf] < osc_vals[prev_osc_conf] and
                osc_vals[prev_osc_conf] >= overbought_threshold):
                bearish[curr_p_conf] = True
                break
            prev_ptr -= 1
    return bearish

@njit(cache=True)
def bullish_divergence(low, osc,
                       price_left_window, price_confirm_bars,
                       osc_left_window, osc_confirm_bars,
                       lookback_bars,
                       oversold_threshold=30.0,
                       min_price_move=0.0, min_osc_move=0.0):
    """
    Bullish divergence: price makes lower low, oscillator makes higher low,
    first oscillator low below oversold_threshold.
    """
    price_pivot, price_vals, _ = detect_swing_lows(low, price_left_window, price_confirm_bars, min_price_move)
    osc_pivot, osc_vals, _ = detect_swing_lows(osc, osc_left_window, osc_confirm_bars, min_osc_move)

    price_idx = np.where(price_pivot)[0]
    osc_idx = np.where(osc_pivot)[0]
    bullish = np.zeros(len(low), dtype=np.bool_)

    if len(price_idx) < 2 or len(osc_idx) < 2:
        return bullish

    for i in range(1, len(price_idx)):
        curr_p_conf = price_idx[i]
        prev_ptr = i - 1
        while prev_ptr >= 0:
            prev_p_conf = price_idx[prev_ptr]
            if curr_p_conf - prev_p_conf > lookback_bars:
                break
            osc_ptr = 0
            while osc_ptr + 1 < len(osc_idx) and osc_idx[osc_ptr + 1] <= curr_p_conf:
                osc_ptr += 1
            if osc_idx[osc_ptr] > curr_p_conf:
                prev_ptr -= 1
                continue
            curr_osc_conf = osc_idx[osc_ptr]

            osc_prev_idx = -1
            for r in range(len(osc_idx)-1, -1, -1):
                if osc_idx[r] <= prev_p_conf:
                    osc_prev_idx = r
                    break
            if osc_prev_idx == -1:
                prev_ptr -= 1
                continue
            prev_osc_conf = osc_idx[osc_prev_idx]

            if (price_vals[curr_p_conf] < price_vals[prev_p_conf] and
                osc_vals[curr_osc_conf] > osc_vals[prev_osc_conf] and
                osc_vals[prev_osc_conf] <= oversold_threshold):
                bullish[curr_p_conf] = True
                break
            prev_ptr -= 1
    return bullish

# =============================================================================
# 4. Ehlers Reflex Divergence class
# =============================================================================
class EhlersReflexDiv:
    """
    Detects bullish/bearish divergences between price and the Ehlers Reflex oscillator.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns 'High', 'Low', and a price source (default 'Close').
    reflex_period : int, default 20
        Lookback for the normalizing MAD.
    reflex_alpha : float, default 0.8
        Smoothing factor of the high‑pass filter (0‑1).
    price_source : str, default 'Close'
        Column name used to compute Reflex.
    price_left_window : int, default 5
    price_confirm_bars : int, default 2
    reflex_left_window : int, default 5
    reflex_confirm_bars : int, default 2
    lookback_bars : int, default 100
    overbought : float or None, default None
    oversold : float or None, default None
    min_price_move : float, default 0.0
    min_reflex_move : float, default 0.0
    """

    def __init__(self, data,
                 reflex_period=20,
                 reflex_alpha=0.8,
                 price_source='Close',
                 price_left_window=3,
                 price_confirm_bars=1,
                 reflex_left_window=3,
                 reflex_confirm_bars=1,
                 lookback_bars=50,
                 overbought=0.9,
                 oversold=0.5,
                 min_price_move=0.0003,
                 min_reflex_move=0.2):
        self.data = data
        self.reflex_period = reflex_period
        self.reflex_alpha = reflex_alpha
        self.price_source = price_source
        self.price_left_window = price_left_window
        self.price_confirm_bars = price_confirm_bars
        self.reflex_left_window = reflex_left_window
        self.reflex_confirm_bars = reflex_confirm_bars
        self.lookback_bars = lookback_bars
        self.overbought = overbought if overbought is not None else -np.inf
        self.oversold   = oversold   if oversold   is not None else  np.inf
        self.min_price_move = min_price_move
        self.min_reflex_move = min_reflex_move

        # Compute Reflex using the njit core
        close = data[price_source].values.astype(np.float64)
        self.reflex_vals = ehlers_reflex_core(close, reflex_period, reflex_alpha)

        # Detect divergences
        self._bearish = bearish_divergence(
            data['High'].values, self.reflex_vals,
            price_left_window, price_confirm_bars,
            reflex_left_window, reflex_confirm_bars,
            lookback_bars,
            overbought_threshold=self.overbought,
            min_price_move=min_price_move,
            min_osc_move=min_reflex_move
        )
        self._bullish = bullish_divergence(
            data['Low'].values, self.reflex_vals,
            price_left_window, price_confirm_bars,
            reflex_left_window, reflex_confirm_bars,
            lookback_bars,
            oversold_threshold=self.oversold,
            min_price_move=min_price_move,
            min_osc_move=min_reflex_move
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
        reflex_series = pd.Series(self.reflex_vals[start_idx:end_idx], index=df_plot.index)

        bearish_plot = self._bearish[start_idx:end_idx]
        bullish_plot = self._bullish[start_idx:end_idx]
        idx_bear = np.where(bearish_plot)[0]
        idx_bull = np.where(bullish_plot)[0]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.6, 0.4],
                            subplot_titles=('Price & Divergences', 'Ehlers Reflex'))
        # Price
        fig.add_trace(go.Candlestick(x=df_plot.index,
                                     open=df_plot['Open'], high=df_plot['High'],
                                     low=df_plot['Low'], close=df_plot['Close'],
                                     name='Price'), row=1, col=1)
        # Bearish markers
        fig.add_trace(go.Scatter(x=df_plot.index[idx_bear],
                                 y=df_plot['High'].iloc[idx_bear] * 1.001,
                                 mode='markers',
                                 marker=dict(color='red', size=12, symbol='arrow-down'),
                                 name='Bearish Div'), row=1, col=1)
        # Bullish markers
        fig.add_trace(go.Scatter(x=df_plot.index[idx_bull],
                                 y=df_plot['Low'].iloc[idx_bull] * 0.999,
                                 mode='markers',
                                 marker=dict(color='lime', size=12, symbol='arrow-up'),
                                 name='Bullish Div'), row=1, col=1)
        # Reflex line
        fig.add_trace(go.Scatter(x=reflex_series.index, y=reflex_series,
                                 mode='lines', line=dict(color='cyan', width=2),
                                 name='Reflex'), row=2, col=1)
        # Zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray",
                      annotation_text="0", row=2, col=1)
        # Thresholds if not infinite
        if self.overbought > -np.inf:
            fig.add_hline(y=self.overbought, line_dash="dash", line_color="red",
                          annotation_text="Overbought", row=2, col=1)
        if self.oversold < np.inf:
            fig.add_hline(y=self.oversold, line_dash="dash", line_color="green",
                          annotation_text="Oversold", row=2, col=1)

        fig.update_layout(title='Ehlers Reflex Divergence',
                          xaxis_title='Date', yaxis_title='Price',
                          height=800, width=1000, template='plotly_dark',
                          hovermode='x unified',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        fig.update_yaxes(title_text="Reflex", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()