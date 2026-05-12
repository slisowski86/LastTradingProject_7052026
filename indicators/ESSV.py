import pandas as pd
import numpy as np
from numba import njit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal

# ----------------------------------------------------------------------
# Numba‑accelerated SuperSmoother filter (unchanged)
# ----------------------------------------------------------------------
@njit(cache=True)
def _supersmoother_core(x, N):
    """Apply John Ehlers' SuperSmoother filter to array x (cutoff period N)."""
    n = len(x)
    if n == 0:
        return np.empty(0)
    out = np.empty(n)
    sqrt2 = np.sqrt(2.0)
    a = np.exp(-sqrt2 * np.pi / N)
    b = 2.0 * a * np.cos(sqrt2 * np.pi / N)
    c = a * a
    c1 = (1.0 - a) * (1.0 - a)
    c2 = b
    c3 = -c

    out[0] = x[0]
    if n > 1:
        out[1] = x[1]
    for i in range(2, n):
        out[i] = c1 * (x[i] + x[i-1]) / 2.0 + c2 * out[i-1] + c3 * out[i-2]
    return out


class ESSV:
    """
    Ehlers SuperSmoother applied to volatility estimate,
    with optional dynamic (rolling percentile) thresholds.

    Regime signals:
        - High volatility when smoothed vol > high_threshold
        - Low  volatility when smoothed vol < low_threshold

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'Close'.
    cutoff_period : int, default 10
        SuperSmoother cutoff period.
    vol_window : int, default 20
        Rolling window for initial volatility estimate (std of returns).
    threshold_mode : str, default 'fixed'
        'fixed' – use a constant scalar threshold (use threshold_high and threshold_low).
        'dynamic' – compute high/low thresholds as rolling percentiles of the smoothed volatility.
    threshold_high : float, default 0.02
        Fixed high threshold when mode='fixed'.
    threshold_low : float, default 0.005
        Fixed low threshold when mode='fixed'.
    dynamic_window : int, default 100
        Lookback window for rolling percentiles when mode='dynamic'.
    dynamic_high_percentile : float, default 75
        Percentile for high‑vol threshold (0-100).
    dynamic_low_percentile : float, default 25
        Percentile for low‑vol threshold.
    """

    def __init__(self, data, cutoff_period=10, vol_window=20,
                 threshold_mode=None,
                 threshold_high=0.000327, threshold_low=0.0002,
                 dynamic_window=100,
                 dynamic_high_percentile=75, dynamic_low_percentile=25):
        self.data = data
        self.cutoff_period = cutoff_period
        self.vol_window = vol_window
        self.threshold_mode = threshold_mode

        # Compute smoothed volatility
        log_ret = np.log(data['Close'] / data['Close'].shift(1))
        self.raw_vol = log_ret.rolling(vol_window).std()
        self.smoothed_vol = self._compute()

        # Build dynamic thresholds if requested
        if threshold_mode == 'dynamic':
            self.high_threshold = self.smoothed_vol.rolling(dynamic_window).apply(
                lambda x: np.percentile(x, dynamic_high_percentile), raw=True
            )
            self.low_threshold = self.smoothed_vol.rolling(dynamic_window).apply(
                lambda x: np.percentile(x, dynamic_low_percentile), raw=True
            )
        else:  # fixed
            self.high_threshold = pd.Series(threshold_high, index=self.data.index)
            self.low_threshold = pd.Series(threshold_low, index=self.data.index)

        self.category = "volatility"

    def _compute(self):
        raw = self.raw_vol.values
        raw = np.where(np.isnan(raw), 0.0, raw)
        smooth = _supersmoother_core(raw, self.cutoff_period)
        return pd.Series(smooth, index=self.data.index, name='SuperSmootherVol')

    @signal(direction="both", signal_type="continuous", weight=1.0)
    def volatility_high(self):
        """1 when smoothed volatility > high_threshold, else 0."""
        return np.where(self.smoothed_vol > self.high_threshold, 1, 0)

    @signal(direction="both", signal_type="continuous", weight=1.0)
    def volatility_medium(self):
        """Return 1 when volatility is between low and high thresholds (inclusive)."""
        return np.where(
            (self.smoothed_vol >= self.low_threshold) &
            (self.smoothed_vol <= self.high_threshold),
            1, 0
        )

    @signal(direction="both", signal_type="continuous", weight=1.0)
    def volatility_low(self):
        """1 when smoothed volatility < low_threshold, else 0."""
        return np.where(self.smoothed_vol < self.low_threshold, 1, 0)

    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]
        vol_plot = self.smoothed_vol.iloc[start_idx:end_idx]
        high_thresh = self.high_threshold.iloc[start_idx:end_idx]
        low_thresh = self.low_threshold.iloc[start_idx:end_idx]

        high_sig = self.volatility_high()[start_idx:end_idx]
        low_sig = self.volatility_low()[start_idx:end_idx]

        idx_high = np.where(high_sig == 1)[0]
        idx_low = np.where(low_sig == 1)[0]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, row_heights=[0.6, 0.4],
            subplot_titles=('Price & Volatility Regimes', 'SuperSmoother Volatility')
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'], high=df_plot['High'],
            low=df_plot['Low'], close=df_plot['Close'],
            name='Price'), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_plot.index[idx_high], y=df_plot['Close'].iloc[idx_high],
            mode='markers', marker=dict(color='orange', size=8, symbol='triangle-up'),
            name='High vol'), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_plot.index[idx_low], y=df_plot['Close'].iloc[idx_low],
            mode='markers', marker=dict(color='green', size=6, symbol='circle'),
            name='Low vol'), row=1, col=1)

        # Volatility panel with thresholds
        fig.add_trace(go.Scatter(
            x=vol_plot.index, y=vol_plot.values,
            mode='lines', line=dict(color='gold', width=2),
            name='Smoothed Vol'), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=high_thresh.index, y=high_thresh.values,
            mode='lines', line=dict(color='red', dash='dash'),
            name='High threshold'), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=low_thresh.index, y=low_thresh.values,
            mode='lines', line=dict(color='green', dash='dot'),
            name='Low threshold'), row=2, col=1)

        fig.update_layout(
            title=f'Ehlers SuperSmoother Volatility (cutoff {self.cutoff_period}, mode={self.threshold_mode})',
            xaxis_title='Date', yaxis_title='Price',
            height=800, width=1000, template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="Volatility", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()