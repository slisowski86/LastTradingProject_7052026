import pandas as pd
import numpy as np
from numba import njit
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal



# ----------------------------------------------------------------------
# Numba-accelerated Fisher Transform core
# ----------------------------------------------------------------------
@njit(cache=True)
def fisher_core(high, low, period):
    """
    Ehlers Fisher Transform on High/Low arrays.
    Returns:
        fisher : array (length n) – Fisher values
        signal : array (length n) – trigger line (Fisher lagged by 1)
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    signal = np.full(n, np.nan)

    if n < period:
        return fisher, signal

    hl2 = (high + low) / 2.0

    # Compute rolling max and min of hl2 over the period
    max_hl2 = np.empty(n)
    min_hl2 = np.empty(n)
    for i in range(period - 1, n):
        window_start = i - period + 1
        max_val = hl2[i]
        min_val = hl2[i]
        for j in range(window_start, i):
            if hl2[j] > max_val:
                max_val = hl2[j]
            if hl2[j] < min_val:
                min_val = hl2[j]
        max_hl2[i] = max_val
        min_hl2[i] = min_val

    # Recursive Fisher transform
    smooth = 0.0
    for i in range(period, n):
        range_hl2 = max_hl2[i] - min_hl2[i]
        if range_hl2 < 1e-10:
            raw = 0.5
        else:
            raw = (hl2[i] - min_hl2[i]) / range_hl2
        position = 2.0 * (raw - 0.5)          # range [-1, +1]

        if i == period:
            smooth = position                # initial seed
        else:
            smooth = 0.33 * position + 0.67 * smooth

        # Clamp to avoid log(0) or infinity
        smooth_clamped = max(min(smooth, 0.999), -0.999)
        fisher[i] = 0.5 * np.log((1.0 + smooth_clamped) / (1.0 - smooth_clamped))

        # Signal = previous Fisher value
        if i > period:
            signal[i] = fisher[i - 1]

    return fisher, signal


# ----------------------------------------------------------------------
# FisherTransform indicator class (same structure as your RSI / CCI)
# ----------------------------------------------------------------------
class EFISH:
    """
    Ehlers Fisher Transform – continuous & discrete signals.

    Continuous  : Fisher > +up_level  (overbought → short)
                  Fisher < down_level (oversold → long)
    Discrete    : Cross above / below zero line
    Category    : momentum

    Parameters
    ----------
    data : pd.DataFrame with 'High','Low','Close'
    period : int, default 10
    up_level : float, default 2.0
    down_level : float, default -2.0
    """
    def __init__(self, data, period=10, up_level=2.0, down_level=-2.0):
        self.data = data
        self.period = period
        self.up_level = up_level
        self.down_level = down_level
        self.fisher, self.fisher_signal = self._compute()
        self.category = "momentum"

    def _compute(self):
        """Use the Numba‑accelerated core and wrap results in pd.Series."""
        high = self.data['High'].values
        low  = self.data['Low'].values
        fisher_arr, signal_arr = fisher_core(high, low, self.period)
        index = self.data.index
        fisher_series = pd.Series(fisher_arr, index=index, name='Fisher')
        signal_series = pd.Series(signal_arr, index=index, name='FisherSignal')
        return fisher_series, signal_series

    # ---------- Continuous overbought / oversold ----------
    @signal(direction="short", signal_type="continuous", weight=1.0)
    def above_up_level_short(self):
        return np.where(self.fisher > self.up_level, -1, 0)

    @signal(direction="long", signal_type="continuous", weight=1.0)
    def below_down_level_long(self):
        return np.where(self.fisher < self.down_level, 1, 0)

    # ---------- Discrete zero‑line crossovers ----------
    @signal(direction="long", signal_type="discrete", weight=2.0)
    def cross_above_zero_long(self):
        prev = self.fisher.shift(1)
        cross = (self.fisher > 0) & (prev <= 0)
        return np.where(cross, 1, 0)

    @signal(direction="short", signal_type="discrete", weight=2.0)
    def cross_below_zero_short(self):
        prev = self.fisher.shift(1)
        cross = (self.fisher < 0) & (prev >= 0)
        return np.where(cross, -1, 0)

    # ---------- Plot (like your RSI/CCI) ----------
    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]
        fisher_plot = self.fisher.iloc[start_idx:end_idx]
        signal_plot = self.fisher_signal.iloc[start_idx:end_idx]

        # Signal arrays
        above_short = self.above_up_level_short()[start_idx:end_idx]
        below_long  = self.below_down_level_long()[start_idx:end_idx]
        cross_above = self.cross_above_zero_long()[start_idx:end_idx]
        cross_below = self.cross_below_zero_short()[start_idx:end_idx]

        idx_above = np.where(above_short == -1)[0]
        idx_below = np.where(below_long  ==  1)[0]
        idx_up    = np.where(cross_above ==  1)[0]
        idx_down  = np.where(cross_below == -1)[0]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, row_heights=[0.6, 0.4],
            subplot_titles=('Price', 'Fisher Transform')
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'], high=df_plot['High'],
            low=df_plot['Low'], close=df_plot['Close'],
            name='Price'
        ), row=1, col=1)

        # Continuous markers (circles)
        fig.add_trace(go.Scatter(
            x=df_plot.index[idx_above],
            y=df_plot['Close'].iloc[idx_above],
            mode='markers',
            marker=dict(color='red', size=8, symbol='circle'),
            name=f'Fisher > +{self.up_level} (short)'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_plot.index[idx_below],
            y=df_plot['Close'].iloc[idx_below],
            mode='markers',
            marker=dict(color='green', size=8, symbol='circle'),
            name=f'Fisher < {self.down_level} (long)'
        ), row=1, col=1)

        # Discrete zero‑line markers (arrows)
        fig.add_trace(go.Scatter(
            x=df_plot.index[idx_down],
            y=df_plot['High'].iloc[idx_down] * 1.0004,
            mode='markers',
            marker=dict(color='red', size=12, symbol='arrow-down'),
            name='Cross below 0 (short)'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_plot.index[idx_up],
            y=df_plot['Low'].iloc[idx_up] * 0.9996,
            mode='markers',
            marker=dict(color='green', size=12, symbol='arrow-up'),
            name='Cross above 0 (long)'
        ), row=1, col=1)

        # Fisher line
        fig.add_trace(go.Scatter(
            x=fisher_plot.index, y=fisher_plot,
            mode='lines', line=dict(color='blue', width=2),
            name='Fisher'
        ), row=2, col=1)

        # Signal (trigger) line
        fig.add_trace(go.Scatter(
            x=signal_plot.index, y=signal_plot,
            mode='lines', line=dict(color='orange', width=1.2, dash='dot'),
            name='Fisher Signal'
        ), row=2, col=1)

        # Reference levels
        fig.add_hline(y=self.up_level, line_dash="dash", line_color="red",
                      annotation_text=f"Overbought (+{self.up_level})", row=2, col=1)
        fig.add_hline(y=self.down_level, line_dash="dash", line_color="green",
                      annotation_text=f"Oversold ({self.down_level})", row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray",
                      annotation_text="Zero", row=2, col=1)

        fig.update_layout(
            title='Fisher Transform Trading Signals',
            xaxis_title='Date', yaxis_title='Price',
            height=800, template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="Fisher Transform", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()