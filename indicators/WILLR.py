import pandas as pd
import numpy as np
from SignalDecorator import signal
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class WILLR:
    """
    Williams %R oscillator – continuous overbought/oversold signals.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'High', 'Low', 'Close' columns.
    up_level : float, default -20
        Overbought threshold (values above this trigger bearish).
    down_level : float, default -80
        Oversold threshold (values below this trigger bullish).
    period : int, default 14
        Lookback period for %R.
    """
    def __init__(self, data, up_level=-20, down_level=-80, period=14):
        self.data = data
        self.up_level = up_level
        self.down_level = down_level
        self.period = period
        self.wpr = self._compute()
        self.category = "momentum"

    def _compute(self):
        """Compute Williams %R using TA‑Lib (scaled 0 to -100)."""
        return ta.WILLR(self.data['High'], self.data['Low'], self.data['Close'],
                        timeperiod=self.period)

    @signal(direction="short", signal_type="continuous", weight=1.0)
    def above_up_level_short(self):
        """-1 when %R > up_level (-20, overbought), else 0."""
        return np.where(self.wpr > self.up_level, -1, 0)

    @signal(direction="long", signal_type="continuous", weight=1.0)
    def below_down_level_long(self):
        """+1 when %R < down_level (-80, oversold), else 0."""
        return np.where(self.wpr < self.down_level, 1, 0)

    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]

        # Prepare signals
        above_short = self.above_up_level_short()[start_idx:end_idx]
        below_long  = self.below_down_level_long()[start_idx:end_idx]

        idx_above = np.where(above_short == -1)[0]
        idx_below = np.where(below_long  ==  1)[0]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=('Price', 'Williams %R')
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'], high=df_plot['High'],
            low=df_plot['Low'], close=df_plot['Close'],
            name='Price'
        ), row=1, col=1)

        # Continuous overbought (short) markers
        fig.add_trace(go.Scatter(
            x=df_plot.index[idx_above],
            y=df_plot['Close'].iloc[idx_above],
            mode='markers',
            marker=dict(color='red', size=8, symbol='circle'),
            name=f'%R > {self.up_level} (short)'
        ), row=1, col=1)

        # Continuous oversold (long) markers
        fig.add_trace(go.Scatter(
            x=df_plot.index[idx_below],
            y=df_plot['Close'].iloc[idx_below],
            mode='markers',
            marker=dict(color='green', size=8, symbol='circle'),
            name=f'%R < {self.down_level} (long)'
        ), row=1, col=1)

        # Williams %R line
        fig.add_trace(go.Scatter(
            x=self.wpr.index[start_idx:end_idx],
            y=self.wpr.iloc[start_idx:end_idx],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Williams %R'
        ), row=2, col=1)

        # Reference lines at -20, -80, -50
        fig.add_hline(y=self.up_level, line_dash="dash", line_color="red",
                      annotation_text=f"Overbought ({self.up_level})", row=2, col=1)
        fig.add_hline(y=self.down_level, line_dash="dash", line_color="green",
                      annotation_text=f"Oversold ({self.down_level})", row=2, col=1)
        fig.add_hline(y=-50, line_dash="dot", line_color="gray", row=2, col=1)

        fig.update_layout(
            title='Williams %R Trading Signals',
            xaxis_title='Date',
            yaxis_title='Price',
            height=800,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="Williams %R", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()