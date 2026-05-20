import pandas as pd
import numpy as np
from numba import njit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SignalDecorator import signal  # your decorator



# ----------------------------------------------------------------------
# Numba-accelerated core computation
# ----------------------------------------------------------------------
@njit(cache=True)
def _eit_core(close, alpha):
    """
    Compute Ehlers Instantaneous Trendline on a 1‑D NumPy array.
    Returns the trendline array (same length as close).
    """
    n = close.shape[0]
    it = np.empty(n)

    if n == 0:
        return it

    # Seed values
    it[0] = close[0]
    if n > 1:
        # Simple 2‑bar EMA for i=1 as seed
        it[1] = alpha * close[1] + (1 - alpha) * it[0]

    # Full recursion from i=2 onwards
    for i in range(2, n):
        term1 = (alpha - alpha**2 / 4.0) * close[i]
        term2 = (alpha**2 / 2.0) * close[i-1]
        term3 = -(alpha - 3.0 * alpha**2 / 4.0) * close[i-2]
        term4 = 2.0 * (1.0 - alpha) * it[i-1]
        term5 = -(1.0 - alpha)**2 * it[i-2]
        it[i] = term1 + term2 + term3 + term4 + term5

    return it


class EIT:
    """
    Ehlers Instantaneous Trendline (EIT) – a zero‑lag smoothing filter.

    Continuous regime signals: slope > 0 → uptrend (1), slope < 0 → downtrend (-1)

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns 'Open', 'High', 'Low', 'Close'.
    cycle_period : int, optional (default 20)
        Used to compute alpha = 2/(cycle_period + 1) if alpha is None.
    alpha : float, optional (default None)
        Smoothing factor. If provided, cycle_period is ignored.
    """

    def __init__(self, data, cycle_period=20, alpha=None):
        self.data = data
        self.cycle_period = cycle_period
        # Use explicit alpha, else derive from cycle_period
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = 2.0 / (cycle_period + 1)
        # The smoothed trendline
        self.eit = self._compute()
        # Slope (first difference)
        self.slope = self.eit.diff()
        self.category = "trend_direction"

    def _compute(self):
        """
        Compute EIT using the numba‑accelerated core and wrap result in a Series.
        """
        close_vals = self.data['Close'].values
        eit_array = _eit_core(close_vals, self.alpha)
        return pd.Series(eit_array, index=self.data.index, name='EIT')

    @signal(direction="long", signal_type="continuous", weight=1.0)
    def uptrend_regime(self):
        """Returns 1 when EIT slope > 0 (uptrend), else 0."""
        return np.where(self.slope > 0, 1, 0)

    @signal(direction="short", signal_type="continuous", weight=1.0)
    def downtrend_regime(self):
        """Returns -1 when EIT slope < 0 (downtrend), else 0."""
        return np.where(self.slope < 0, -1, 0)

    def plot(self, start_idx=None, end_idx=None):
        """
        Interactive Plotly chart: price candlesticks + regime markers,
        and the EIT line in a second panel.
        """
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]
        eit_plot = self.eit.iloc[start_idx:end_idx]

        # Signal arrays for the plotted range
        uptrend = self.uptrend_regime()[start_idx:end_idx]
        downtrend = self.downtrend_regime()[start_idx:end_idx]

        idx_up = np.where(uptrend == 1)[0]
        idx_down = np.where(downtrend == -1)[0]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=('Price & Regime Markers', 'Ehlers Instantaneous Trendline')
        )

        # ---------- Row 1: Candlestick + regime markers ----------
        fig.add_trace(
            go.Candlestick(
                x=df_plot.index,
                open=df_plot['Open'],
                high=df_plot['High'],
                low=df_plot['Low'],
                close=df_plot['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Uptrend markers (green circles)
        fig.add_trace(
            go.Scatter(
                x=df_plot.index[idx_up],
                y=df_plot['Close'].iloc[idx_up],
                mode='markers',
                marker=dict(color='green', size=8, symbol='circle'),
                name='Uptrend (slope > 0)'
            ),
            row=1, col=1
        )

        # Downtrend markers (red circles)
        fig.add_trace(
            go.Scatter(
                x=df_plot.index[idx_down],
                y=df_plot['Close'].iloc[idx_down],
                mode='markers',
                marker=dict(color='red', size=8, symbol='circle'),
                name='Downtrend (slope < 0)'
            ),
            row=1, col=1
        )

        # ---------- Row 2: EIT line ----------
        fig.add_trace(
            go.Scatter(
                x=eit_plot.index,
                y=eit_plot.values,
                mode='lines',
                line=dict(color='gold', width=2),
                name='EIT'
            ),
            row=2, col=1
        )

        # Layout
        fig.update_layout(
            title='Ehlers Instantaneous Trendline – Regime Signals',
            xaxis_title='Date',
            yaxis_title='Price',
            height=800,
            width=1000,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="EIT Value", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()