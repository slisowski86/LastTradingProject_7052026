import pandas as pd 
import numpy as np
from SignalDecorator import *
import talib as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots



class GarmanKlass:
    def __init__(self, data, window=20,
                 threshold_high=0.01, threshold_low=0.005,   # new dual thresholds
                 threshold=None,                           # legacy single
                 dynamic_threshold='percentile',
                 percentile_high=70, percentile_low=30,
                 std_mult_high=1.0, std_mult_low=1.0,
                 long_window=252, annualize=True, years_per_period=252):
        self.data = data
        self.window = window
        self.annualize = annualize
        self.years_per_period = years_per_period
        self.gk_vol = self._compute()
        self.category = "volatility"

        # ---- Decide threshold mode ----
        if threshold_high is not None or threshold_low is not None:
            self.threshold_high = threshold_high
            self.threshold_low = threshold_low
            self.threshold_type = 'static_dual'
        elif threshold is not None:
            self.threshold_high = threshold
            self.threshold_low = threshold
            self.threshold_type = 'static'
        else:
            self.threshold_type = dynamic_threshold
            self.percentile_high = percentile_high
            self.percentile_low = percentile_low
            self.std_mult_high = std_mult_high
            self.std_mult_low = std_mult_low
            self.long_window = long_window
            self._update_dynamic_threshold()

    def _compute(self):
        high_low = np.log(self.data['High'] / self.data['Low'])
        close_open = np.log(self.data['Close'] / self.data['Open'])
        gk_var = 0.5 * high_low**2 - (2 * np.log(2) - 1) * close_open**2
        mean_var = gk_var.rolling(window=self.window).mean()
        if self.annualize:
            vol = np.sqrt(mean_var * self.years_per_period)
        else:
            vol = np.sqrt(mean_var)
        return vol

    def _update_dynamic_threshold(self):
        """Compute two dynamic threshold series (high & low)."""
        if self.threshold_type == 'percentile':
            self.dynamic_high = self.gk_vol.rolling(self.long_window).quantile(
                self.percentile_high / 100.0)
            self.dynamic_low  = self.gk_vol.rolling(self.long_window).quantile(
                self.percentile_low / 100.0)
        elif self.threshold_type == 'std':
            rolling_mean = self.gk_vol.rolling(self.long_window).mean()
            rolling_std  = self.gk_vol.rolling(self.long_window).std()
            self.dynamic_high = rolling_mean + self.std_mult_high * rolling_std
            self.dynamic_low  = rolling_mean - self.std_mult_low * rolling_std
        elif self.threshold_type == 'mean':
            self.dynamic_high = self.gk_vol.rolling(self.long_window).mean()
            self.dynamic_low  = self.dynamic_high   # fallback
        else:
            raise ValueError("Invalid dynamic_threshold")

    @signal(direction="both", signal_type="continuous", weight=1.0)
    def high_volatility_regime(self):
        """Return 1 when volatility > high threshold."""
        if self.threshold_type in ('static_dual', 'static'):
            thr = self.threshold_high
        else:
            thr = self.dynamic_high
        return np.where(self.gk_vol > thr, 1, 0)

    @signal(direction="both", signal_type="continuous", weight=1.0)
    def medium_volatility_regime(self):
        """Return 1 when volatility is between low and high thresholds (inclusive)."""
        if self.threshold_type in ('static_dual', 'static'):
            high = self.threshold_high
            low  = self.threshold_low
        else:
            high = self.dynamic_high
            low  = self.dynamic_low
        return np.where((self.gk_vol >= low) & (self.gk_vol <= high), 1, 0)

    @signal(direction="both", signal_type="continuous", weight=1.0)
    def low_volatility_regime(self):
        """Return 1 when volatility < low threshold."""
        if self.threshold_type in ('static_dual', 'static'):
            thr = self.threshold_low
        else:
            thr = self.dynamic_low
        return np.where(self.gk_vol < thr, 1, 0)

    def plot(self, start_idx=None, end_idx=None):
        """Interactive Plotly chart with dual thresholds & regime markers."""
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)

        df_plot = self.data.iloc[start_idx:end_idx]
        vol_plot = self.gk_vol.iloc[start_idx:end_idx]

        # --- Get threshold series ---
        if self.threshold_type == 'static_dual' or self.threshold_type == 'static':
            high_thr = pd.Series(self.threshold_high, index=vol_plot.index)
            low_thr  = pd.Series(self.threshold_low, index=vol_plot.index)
        else:
            high_thr = self.dynamic_high.iloc[start_idx:end_idx]
            low_thr  = self.dynamic_low.iloc[start_idx:end_idx]

        # --- Signal arrays for markers ---
        high_sig = self.high_volatility_regime()[start_idx:end_idx]
        low_sig  = self.low_volatility_regime()[start_idx:end_idx]
        idx_high = np.where(high_sig == 1)[0]
        idx_low  = np.where(low_sig == 1)[0]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, row_heights=[0.6, 0.4],
            subplot_titles=('Price & Volatility Regimes', 'Garman‑Klass Volatility')
        )

        # ---- Row 1: Candlestick + regime markers ----
        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'], high=df_plot['High'],
            low=df_plot['Low'], close=df_plot['Close'],
            name='Price'
        ), row=1, col=1)

        # High volatility marker (orange triangle)
        fig.add_trace(go.Scatter(
            x=df_plot.index[idx_high],
            y=df_plot['Close'].iloc[idx_high],
            mode='markers',
            marker=dict(color='orange', size=9, symbol='triangle-up'),
            name='High Volatility'
        ), row=1, col=1)

        # Low volatility marker (green circle)
        fig.add_trace(go.Scatter(
            x=df_plot.index[idx_low],
            y=df_plot['Close'].iloc[idx_low],
            mode='markers',
            marker=dict(color='green', size=7, symbol='circle'),
            name='Low Volatility'
        ), row=1, col=1)

        # ---- Row 2: Volatility line with colored segments ----
        # Build a color for each segment based on both thresholds
        colors = []
        for i in range(len(vol_plot)):
            v = vol_plot.iloc[i]
            if v > high_thr.iloc[i]:
                colors.append('red')        # above high -> red (high risk)
            elif v < low_thr.iloc[i]:
                colors.append('green')      # below low -> green (calm)
            else:
                colors.append('orange')    # between -> orange (normal)

        # Plot as individual segments (show only one legend entry for the line)
        for i in range(1, len(vol_plot)):
            fig.add_trace(go.Scatter(
                x=vol_plot.index[i-1:i+1],
                y=vol_plot.iloc[i-1:i+1],
                mode='lines',
                line=dict(color=colors[i-1], width=2),
                showlegend=False
            ), row=2, col=1)

        # Add a dummy trace for the legend
        fig.add_trace(go.Scatter(
            x=[vol_plot.index[0]], y=[vol_plot.iloc[0]],
            mode='lines', line=dict(color='darkgray', width=2),
            name='GK Volatility'
        ), row=2, col=1)

        # Threshold lines
        fig.add_trace(go.Scatter(
            x=high_thr.index, y=high_thr.values,
            mode='lines', line=dict(color='red', dash='dash', width=1),
            name='High Threshold'
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=low_thr.index, y=low_thr.values,
            mode='lines', line=dict(color='green', dash='dot', width=1),
            name='Low Threshold'
        ), row=2, col=1)

        fig.update_layout(
            title='Garman‑Klass Volatility with Dual Thresholds',
            xaxis_title='Date', yaxis_title='Price',
            height=800, width=1000, template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.update_yaxes(title_text="Volatility (decimal)", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()