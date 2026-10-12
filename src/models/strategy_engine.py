"""
Quantitative Macro Trading Strategy Engine.
Simulates multiple macro strategy permutations and computes transaction-cost-adjusted
daily P&L, equity curves, and trade win rates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from config.settings import TRANSACTION_COST_BPS, RISK_FREE_RATE

class MacroStrategyEngine:
    def __init__(self, cost_bps: float = TRANSACTION_COST_BPS):
        self.cost_rate = cost_bps / 10000.0  # 5 bps = 0.0005

    def backtest_strategies(self, df_features: pd.DataFrame, predicted_signals: pd.Series) -> pd.DataFrame:
        """
        Backtests 5 distinct macro strategy permutations:
        1. Macro Surprise Momentum (Long/Short equities on CPI/NFP surprises)
        2. Fed Policy Drift (Long equities & bonds post-FOMC cuts)
        3. Pre-Event Volatility Breakout (Leverage during low-vol pre-FOMC windows)
        4. Cross-Asset Regime Parity (Multi-asset risk allocation: Equities + Gold + Bonds)
        5. ML-Guided Dynamic Macro (Position sizing modulated by ML model confidence)
        """
        results = pd.DataFrame(index=df_features.index)
        
        spy_ret = df_features["spy_ret_1d"].values
        tlt_ret = df_features.get("tlt_ret_5d", pd.Series(0, index=df_features.index)).values / 5.0
        gld_ret = df_features.get("gld_ret_5d", pd.Series(0, index=df_features.index)).values / 5.0

        cpi_surprise = df_features["latest_cpi_surprise"].values
        fomc_decay = df_features["fomc_decay"].values
        is_pre_fomc = df_features["is_pre_fomc"].values
        sentiment = df_features["rolling_macro_sentiment"].values
        ml_signal = predicted_signals.reindex(df_features.index).fillna(0.0).values

        n = len(df_features)

        # 1. Macro Surprise Momentum:
        # Negative CPI surprise (disinflation) -> Long Equities. Positive CPI surprise (hawkish) -> Short / Cash.
        weights_strat_1 = np.where(cpi_surprise <= 0, 1.0, -0.5)
        # Apply transaction cost on position changes
        trades_1 = np.abs(np.diff(weights_strat_1, prepend=0.0))
        ret_strat_1 = (weights_strat_1 * spy_ret) - (trades_1 * self.cost_rate)
        results["strat_cpi_momentum"] = ret_strat_1

        # 2. Fed Policy Drift:
        # High exposure post-FOMC rate cuts, blended across SPY and TLT
        weights_strat_2 = np.clip(1.0 + fomc_decay, 0.5, 1.5)
        trades_2 = np.abs(np.diff(weights_strat_2, prepend=0.0))
        ret_strat_2 = (weights_strat_2 * (0.6 * spy_ret + 0.4 * tlt_ret)) - (trades_2 * self.cost_rate)
        results["strat_fed_drift"] = ret_strat_2

        # 3. Volatility Breakout & Event Straddle:
        # Underweight before FOMC, double leverage on confirmed breakout after
        weights_strat_3 = np.where(is_pre_fomc == 1, 0.2, np.where(sentiment > 0.4, 1.25, 0.75))
        trades_3 = np.abs(np.diff(weights_strat_3, prepend=0.0))
        ret_strat_3 = (weights_strat_3 * spy_ret) - (trades_3 * self.cost_rate)
        results["strat_vol_breakout"] = ret_strat_3

        # 4. Cross-Asset Regime Parity:
        # Balanced all-weather macro allocation (40% SPY, 30% TLT, 30% GLD)
        ret_strat_4 = (0.40 * spy_ret) + (0.30 * tlt_ret) + (0.30 * gld_ret)
        results["strat_regime_parity"] = ret_strat_4

        # 5. ML-Guided Dynamic Macro:
        # Modulates exposure based on out-of-sample ML model confidence and macro regime
        # Leverages risk-on conditions up to 1.25x while de-risking to 0.25x in volatile/pre-event periods
        ml_regime_score = ml_signal + (0.4 * sentiment) - (0.3 * np.abs(cpi_surprise))
        weights_strat_5 = np.where(
            is_pre_fomc == 1, 0.25,
            np.where(ml_regime_score > 0.05, 1.25,
            np.where(ml_regime_score < -0.05, 0.20, 0.80))
        )
        trades_5 = np.abs(np.diff(weights_strat_5, prepend=0.0))
        ret_strat_5 = (weights_strat_5 * spy_ret) - (trades_5 * self.cost_rate)
        results["strat_ml_dynamic_macro"] = ret_strat_5

        # Benchmark: SPY Buy & Hold
        results["benchmark_spy"] = spy_ret

        return results

    @staticmethod
    def calculate_metrics(returns: pd.Series, rf_annual: float = RISK_FREE_RATE) -> Dict[str, float]:
        """Calculates institutional-grade risk and performance metrics."""
        returns = returns.dropna()
        if len(returns) == 0 or returns.std() == 0:
            return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "cagr": 0.0}

        daily_rf = rf_annual / 252.0
        excess_returns = returns - daily_rf
        
        # Annualized Mean & Volatility
        mean_excess = excess_returns.mean() * 252.0
        annual_vol = returns.std() * np.sqrt(252.0)
        
        # Sharpe Ratio
        sharpe = round(mean_excess / annual_vol, 3) if annual_vol > 0 else 0.0

        # Sortino Ratio (Downside deviation only)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252.0) if len(downside_returns) > 0 else 1e-4
        sortino = round(mean_excess / downside_vol, 3) if downside_vol > 0 else 0.0

        # Cumulative Wealth & Maximum Drawdown
        cum_wealth = (1.0 + returns).cumprod()
        peak = cum_wealth.cummax()
        drawdowns = (cum_wealth - peak) / peak
        max_drawdown = round(abs(drawdowns.min()) * 100.0, 2)

        # Win Rate (%)
        win_rate = round((returns > 0).mean() * 100.0, 2)

        # Profit Factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999.0

        # Total Cumulative Return (%)
        total_return = round((cum_wealth.iloc[-1] - 1.0) * 100.0, 2)

        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown_pct": max_drawdown,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "total_return_pct": total_return
        }
