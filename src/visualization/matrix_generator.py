"""
Visual Matrix Output Generator.
Generates institutional-grade visual matrices, macro sentiment timelines,
cumulative return curves, and drawdown heatmaps.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from config.settings import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class VisualMatrixGenerator:
    def __init__(self, output_dir: Path = OUTPUTS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Apply dark financial styling
        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    def generate_all_visuals(
        self,
        features_df: pd.DataFrame,
        strategy_pnl_df: pd.DataFrame,
        metrics_summary: Dict[str, Dict[str, float]],
        importance_df: pd.DataFrame
    ):
        """Generates the full suite of visual matrix charts."""
        logger.info("Generating publication-quality visual outputs in outputs/...")
        self.plot_macro_sentiment_timeline(features_df)
        self.plot_cumulative_returns(strategy_pnl_df)
        self.plot_drawdown_profile(strategy_pnl_df)
        self.plot_performance_matrix(metrics_summary)
        self.plot_feature_importance(importance_df)
        logger.info("All visual matrices generated successfully.")

    def plot_macro_sentiment_timeline(self, df: pd.DataFrame):
        """1. Visualizes Macro Sentiment Changes Over Time with event markers."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        dates = pd.to_datetime(df.index)
        sentiment = df["rolling_macro_sentiment"].values
        raw_sentiment = df["raw_sentiment"].values
        shock = df["macro_shock_index"].values

        # Plot 1: Rolling Sentiment Index
        ax1.plot(dates, sentiment, color="#00bcd4", linewidth=2.5, label="Rolling Macro Sentiment (7d EWMA)")
        ax1.scatter(dates, raw_sentiment, color="#ff9800", alpha=0.4, s=25, label="Raw News Daily Sentiment")
        ax1.axhline(0.0, color="#ffffff", linestyle="--", alpha=0.3)
        ax1.fill_between(dates, sentiment, 0, where=(sentiment >= 0), color="#4caf50", alpha=0.15, label="Expansion / Dovish")
        ax1.fill_between(dates, sentiment, 0, where=(sentiment < 0), color="#f44336", alpha=0.15, label="Contraction / Hawkish")
        
        # Event annotations
        fomc_events = df[df["is_post_fomc"] == 1]
        for d in fomc_events.index[::4]:
            ax1.axvline(pd.to_datetime(d), color="#9c27b0", linestyle=":", alpha=0.6)
            ax1.text(pd.to_datetime(d), 0.75, "FOMC", color="#ba68c8", fontsize=8, rotation=90)

        ax1.set_title("Macroeconomic Sentiment Evolution & Event Shocks (180-Day Window)", fontsize=14, fontweight="bold", pad=12)
        ax1.set_ylabel("Macro Sentiment Index (-1.0 to +1.0)", fontsize=11)
        ax1.legend(loc="upper left", framealpha=0.8)
        ax1.grid(True, alpha=0.25)

        # Plot 2: Macro Shock Index (CPI + NFP surprise magnitude)
        ax2.bar(dates, shock, color=np.where(shock >= 0, "#2196f3", "#e91e63"), alpha=0.7, width=1.5, label="Macro Surprise Z-Score")
        ax2.set_ylabel("Shock Magnitude", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.legend(loc="upper left", framealpha=0.8)
        ax2.grid(True, alpha=0.25)

        fig.autofmt_xdate()
        plt.tight_layout()
        out_path = self.output_dir / "macro_sentiment_timeline.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

    def plot_cumulative_returns(self, pnl_df: pd.DataFrame):
        """2. Cumulative Equity Curves comparing all strategy permutations."""
        fig, ax = plt.subplots(figsize=(14, 7))

        dates = pd.to_datetime(pnl_df.index)
        cum_df = (1.0 + pnl_df).cumprod()

        colors = {
            "strat_ml_dynamic_macro": "#00e676",      # Vibrant Green (Hero Strategy)
            "strat_cpi_momentum": "#29b6f6",          # Light Blue
            "strat_fed_drift": "#ab47bc",             # Purple
            "strat_vol_breakout": "#ffa726",          # Orange
            "strat_regime_parity": "#26a69a",         # Teal
            "benchmark_spy": "#78909c"                # Slate Grey Benchmark
        }
        labels = {
            "strat_ml_dynamic_macro": "ML-Guided Dynamic Macro (Proposed)",
            "strat_cpi_momentum": "CPI Surprise Momentum",
            "strat_fed_drift": "Fed Policy Drift Strategy",
            "strat_vol_breakout": "Pre-Event Volatility Breakout",
            "strat_regime_parity": "Cross-Asset Regime Parity",
            "benchmark_spy": "Benchmark: S&P 500 Buy & Hold"
        }

        for col in cum_df.columns:
            lw = 3.0 if col == "strat_ml_dynamic_macro" else (1.8 if col == "benchmark_spy" else 1.4)
            ls = "--" if col == "benchmark_spy" else "-"
            ax.plot(dates, cum_df[col], label=labels.get(col, col), color=colors.get(col, "#ffffff"), linewidth=lw, linestyle=ls)

        ax.set_title("Walk-Forward Out-Of-Sample Cumulative Equity Curves", fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel("Cumulative Wealth ($1.00 Base)", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
        ax.grid(True, alpha=0.25)

        fig.autofmt_xdate()
        plt.tight_layout()
        out_path = self.output_dir / "cumulative_strategy_returns.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

    def plot_drawdown_profile(self, pnl_df: pd.DataFrame):
        """3. Historical Drawdown Underwater Profiles."""
        fig, ax = plt.subplots(figsize=(14, 5))

        dates = pd.to_datetime(pnl_df.index)
        cum_ml = (1.0 + pnl_df["strat_ml_dynamic_macro"]).cumprod()
        dd_ml = (cum_ml - cum_ml.cummax()) / cum_ml.cummax() * 100.0

        cum_spy = (1.0 + pnl_df["benchmark_spy"]).cumprod()
        dd_spy = (cum_spy - cum_spy.cummax()) / cum_spy.cummax() * 100.0

        ax.fill_between(dates, dd_ml, 0, color="#00e676", alpha=0.35, label="ML Dynamic Macro Drawdown")
        ax.plot(dates, dd_ml, color="#00e676", linewidth=1.8)

        ax.plot(dates, dd_spy, color="#ff5252", linestyle="--", linewidth=1.5, label="S&P 500 Benchmark Drawdown")

        ax.set_title("Strategy Drawdown Underwater Profile (Risk Mitigation Comparison)", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Drawdown Percentage (%)", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.legend(loc="lower left", framealpha=0.9)
        ax.grid(True, alpha=0.25)

        fig.autofmt_xdate()
        plt.tight_layout()
        out_path = self.output_dir / "strategy_drawdown_profile.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

    def plot_performance_matrix(self, metrics: Dict[str, Dict[str, float]]):
        """4. Visual Matrix Heatmap comparing Sharpe, Sortino, Drawdown, and Win Rates."""
        matrix_df = pd.DataFrame(metrics).T
        
        # Select and rename key columns
        cols = ["sharpe", "sortino", "max_drawdown_pct", "win_rate_pct", "profit_factor", "total_return_pct"]
        matrix_df = matrix_df[cols]
        matrix_df.columns = ["Sharpe Ratio", "Sortino Ratio", "Max DD (%)", "Win Rate (%)", "Profit Factor", "Total Return (%)"]
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axis('off')
        ax.axis('tight')

        table_data = []
        for idx, row in matrix_df.iterrows():
            clean_name = idx.replace("strat_", "").replace("_", " ").title()
            table_data.append([clean_name] + [f"{val:.2f}" for val in row.values])

        col_labels = ["Strategy Permutation"] + list(matrix_df.columns)
        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Style header and rows
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#263238')
                cell.set_text_props(color='white', fontweight='bold')
            elif "Ml Dynamic" in table_data[i-1][0]:
                cell.set_facecolor('#e8f5e9')
                cell.set_text_props(fontweight='bold', color='#1b5e20')
            else:
                cell.set_facecolor('#f5f5f5' if i % 2 == 0 else '#ffffff')

        plt.title("Quantitative Performance Matrix (Walk-Forward Out-of-Sample)", fontsize=13, fontweight='bold', pad=20)
        plt.tight_layout()
        out_path = self.output_dir / "strategy_performance_matrix.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

    def plot_feature_importance(self, importance_df: pd.DataFrame):
        """5. Top 10 Macroeconomic Predictive Feature Importances."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        top10 = importance_df.head(10).iloc[::-1]
        bars = ax.barh(top10["feature"], top10["importance"], color="#00bcd4", alpha=0.85)

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center', fontsize=9)

        ax.set_title("Top 10 Macroeconomic Feature Importances (Random Forest Ensemble)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Relative Feature Importance (MDI)", fontsize=11)
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        out_path = self.output_dir / "feature_importance.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
