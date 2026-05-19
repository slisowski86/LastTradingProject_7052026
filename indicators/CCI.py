
import pandas as pd 
import numpy as np
from SignalDecorator import *
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class CCI:
    """
    Commodity Channel Index (CCI) with continuous signals at ±100
    and discrete zero‑line crossovers.
    """
    def __init__(self, data, up_level=100, down_level=-100, period=20):
        self.data = data
        self.up_level = up_level
        self.down_level = down_level
        self.period = period
        self.cci = self._compute()
        self.category = "momentum"

    def _compute(self):
        return ta.CCI(self.data['High'], self.data['Low'], self.data['Close'],
                      timeperiod=self.period)

    # ---------- Continuous overbought / oversold ----------
    @signal(direction="short", signal_type="continuous", weight=1.0)
    def above_up_level_short(self):
        """-1 when CCI > +100 (overbought), else 0."""
        return np.where(self.cci > self.up_level, -1, 0)

    @signal(direction="long", signal_type="continuous", weight=1.0)
    def below_down_level_long(self):
        """+1 when CCI < -100 (oversold), else 0."""
        return np.where(self.cci < self.down_level, 1, 0)

    # ---------- Discrete zero‑line crossovers ----------
    @signal(direction="long", signal_type="discrete", weight=2.0)
    def cross_above_zero_long(self):
        """+1 when CCI crosses above 0 (previous <= 0, current > 0)."""
        prev = self.cci.shift(1)
        cross = (self.cci > 0) & (prev <= 0)
        return np.where(cross, 1, 0)

    @signal(direction="short", signal_type="discrete", weight=2.0)
    def cross_below_zero_short(self):
        """-1 when CCI crosses below 0 (previous >= 0, current < 0)."""
        prev = self.cci.shift(1)
        cross = (self.cci < 0) & (prev >= 0)
        return np.where(cross, -1, 0)

    def plot(self, start_idx=None, end_idx=None):
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]

        # Signals
        above_short = self.above_up_level_short()[start_idx:end_idx]
        below_long  = self.below_down_level_long()[start_idx:end_idx]
        cross_above = self.cross_above_zero_long()[start_idx:end_idx]
        cross_below = self.cross_below_zero_short()[start_idx:end_idx]

        idx_above_short = np.where(above_short == -1)[0]
        idx_below_long  = np.where(below_long  ==  1)[0]
        idx_cross_above = np.where(cross_above ==  1)[0]
        idx_cross_below = np.where(cross_below == -1)[0]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=('Price', 'CCI')
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
            x=df_plot.index[idx_above_short],
            y=df_plot['Close'].iloc[idx_above_short],
            mode='markers',
            marker=dict(color='red', size=8, symbol='circle'),
            name='CCI > +100 (short)'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_plot.index[idx_below_long],
            y=df_plot['Close'].iloc[idx_below_long],
            mode='markers',
            marker=dict(color='green', size=8, symbol='circle'),
            name='CCI < -100 (long)'
        ), row=1, col=1)

        # Discrete zero‑line markers (arrows)
        fig.add_trace(go.Scatter(
            x=df_plot.index[idx_cross_below],
            y=df_plot['High'].iloc[idx_cross_below] * 1.0004,
            mode='markers',
            marker=dict(color='red', size=12, symbol='arrow-down'),
            name='Cross below 0 (short)'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_plot.index[idx_cross_above],
            y=df_plot['Low'].iloc[idx_cross_above] * 0.9996,
            mode='markers',
            marker=dict(color='green', size=12, symbol='arrow-up'),
            name='Cross above 0 (long)'
        ), row=1, col=1)

        # CCI line
        fig.add_trace(go.Scatter(
            x=self.cci.index[start_idx:end_idx],
            y=self.cci.iloc[start_idx:end_idx],
            mode='lines',
            line=dict(color='blue', width=2),
            name='CCI'
        ), row=2, col=1)

        # Reference lines
        fig.add_hline(y=self.up_level, line_dash="dash", line_color="red",
                      annotation_text=f"Overbought (+{self.up_level})", row=2, col=1)
        fig.add_hline(y=self.down_level, line_dash="dash", line_color="green",
                      annotation_text=f"Oversold ({self.down_level})", row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray",
                      annotation_text="Zero", row=2, col=1)

        fig.update_layout(
            title='CCI Trading Signals (±100 levels & Zero‑line Crossovers)',
            xaxis_title='Date',
            yaxis_title='Price',
            height=800,
            width=1000,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="CCI", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()