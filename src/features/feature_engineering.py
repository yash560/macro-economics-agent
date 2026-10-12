"""
Quantitative Feature Engineering Module.
Constructs time-based event features, macroeconomic surprise z-scores,
rolling sentiment indices, cross-asset momentum, and macro regime classifications.
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from config.settings import EVENT_TYPES, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class MacroFeatureEngine:
    def __init__(self):
        pass

    def build_features(
        self,
        events_df: pd.DataFrame,
        news_df: pd.DataFrame,
        prices_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges events, news sentiment, and asset price returns into a cohesive daily feature matrix.
        Strictly prevents lookahead bias by using backward-looking or contemporaneously available data.
        """
        logger.info("Building quantitative macroeconomic feature matrix...")
        
        # 1. Pivot price data by symbol
        pivot_prices = prices_df.pivot(index="date", columns="symbol", values="adj_close")
        pivot_returns = prices_df.pivot(index="date", columns="symbol", values="daily_return")
        
        all_dates = sorted(pivot_prices.index.unique())
        feature_rows = []

        # Convert events dates for lookup
        events_df["event_date"] = pd.to_datetime(events_df["date"])
        news_df["news_date"] = pd.to_datetime(news_df["date"])

        # Extract event-specific dates
        fomc_dates = pd.to_datetime(events_df[events_df["event_type"] == "FOMC_RATE_DECISION"]["date"]).sort_values().values
        cpi_dates = pd.to_datetime(events_df[events_df["event_type"] == "CPI_INFLATION"]["date"]).sort_values().values
        nfp_dates = pd.to_datetime(events_df[events_df["event_type"] == "NON_FARM_PAYROLLS"]["date"]).sort_values().values

        # Aggregate daily news sentiment
        daily_sentiment = news_df.groupby("date")["sentiment_score"].mean().to_dict()

        for d_str in all_dates:
            dt = pd.to_datetime(d_str)

            # --- Time-Based Event Features ---
            # Distance to past and future FOMC meetings
            past_fomc = [f for f in fomc_dates if f <= dt]
            future_fomc = [f for f in fomc_dates if f > dt]

            days_since_fomc = (dt - past_fomc[-1]).days if len(past_fomc) > 0 else 60
            days_until_fomc = (future_fomc[0] - dt).days if len(future_fomc) > 0 else 60

            # Distance to past CPI and NFP releases
            past_cpi = [c for c in cpi_dates if c <= dt]
            past_nfp = [n for n in nfp_dates if n <= dt]

            days_since_cpi = (dt - past_cpi[-1]).days if len(past_cpi) > 0 else 30
            days_since_nfp = (dt - past_nfp[-1]).days if len(past_nfp) > 0 else 30

            # Decay factor for macro shock absorption (half-life of ~4 days)
            fomc_decay = float(np.exp(-days_since_fomc / 4.0))
            cpi_decay = float(np.exp(-days_since_cpi / 4.0))
            nfp_decay = float(np.exp(-days_since_nfp / 4.0))

            # Pre-event blackout/anticipation flags (within 3 business days before meeting)
            is_pre_fomc = 1 if days_until_fomc <= 3 else 0
            is_post_fomc = 1 if days_since_fomc <= 3 else 0

            # --- Macroeconomic Surprises (Carry forward latest known release) ---
            known_events = events_df[events_df["event_date"] <= dt]
            
            cpi_sub = known_events[known_events["event_type"] == "CPI_INFLATION"]
            latest_cpi_surprise = float(cpi_sub.iloc[-1]["surprise_std"]) if len(cpi_sub) > 0 else 0.0

            nfp_sub = known_events[known_events["event_type"] == "NON_FARM_PAYROLLS"]
            latest_nfp_surprise = float(nfp_sub.iloc[-1]["surprise_std"]) if len(nfp_sub) > 0 else 0.0

            fomc_sub = known_events[known_events["event_type"] == "FOMC_RATE_DECISION"]
            latest_rate = float(fomc_sub.iloc[-1]["actual"]) if len(fomc_sub) > 0 else 5.25

            # Composite macro shock metric
            macro_shock_index = round(
                (latest_cpi_surprise * cpi_decay) + (latest_nfp_surprise * nfp_decay * 0.5), 4
            )

            # Raw news sentiment score (default neutral is 0.0)
            raw_sentiment = daily_sentiment.get(d_str, 0.0)

            feature_rows.append({
                "date": d_str,
                "days_since_fomc": days_since_fomc,
                "days_until_fomc": days_until_fomc,
                "days_since_cpi": days_since_cpi,
                "days_since_nfp": days_since_nfp,
                "fomc_decay": fomc_decay,
                "cpi_decay": cpi_decay,
                "nfp_decay": nfp_decay,
                "is_pre_fomc": is_pre_fomc,
                "is_post_fomc": is_post_fomc,
                "latest_cpi_surprise": latest_cpi_surprise,
                "latest_nfp_surprise": latest_nfp_surprise,
                "current_fed_rate": latest_rate,
                "macro_shock_index": macro_shock_index,
                "raw_sentiment": raw_sentiment
            })

        feat_df = pd.DataFrame(feature_rows)
        feat_df.set_index("date", inplace=True)

        # --- Rolling Market & Cross-Asset Features ---
        spy_ret = pivot_returns.get("SPY", pd.Series(0, index=pivot_returns.index))
        tlt_ret = pivot_returns.get("TLT", pd.Series(0, index=pivot_returns.index))
        gld_ret = pivot_returns.get("GLD", pd.Series(0, index=pivot_returns.index))
        uup_ret = pivot_returns.get("UUP", pd.Series(0, index=pivot_returns.index))

        feat_df["spy_ret_1d"] = spy_ret
        feat_df["spy_ret_5d"] = (1.0 + spy_ret).rolling(5).apply(np.prod, raw=True) - 1.0
        feat_df["tlt_ret_5d"] = (1.0 + tlt_ret).rolling(5).apply(np.prod, raw=True) - 1.0
        feat_df["gld_ret_5d"] = (1.0 + gld_ret).rolling(5).apply(np.prod, raw=True) - 1.0
        feat_df["uup_ret_5d"] = (1.0 + uup_ret).rolling(5).apply(np.prod, raw=True) - 1.0

        # Rolling 20-day annualized realized volatility (252 trading days)
        feat_df["spy_vol_20d"] = spy_ret.rolling(20).std() * np.sqrt(252)
        feat_df["tlt_vol_20d"] = tlt_ret.rolling(20).std() * np.sqrt(252)

        # Yield curve proxy / Risk asset momentum (SPY / TLT relative strength)
        spy_prices = pivot_prices["SPY"]
        tlt_prices = pivot_prices["TLT"]
        feat_df["spy_tlt_ratio"] = spy_prices / tlt_prices
        feat_df["yield_curve_slope"] = feat_df["spy_tlt_ratio"].pct_change(10)

        # Smoothed EWMA Macro Sentiment
        feat_df["rolling_macro_sentiment"] = feat_df["raw_sentiment"].ewm(span=7).mean()

        # Target forward returns for model prediction (1-day and 5-day out-of-sample forward return)
        feat_df["target_spy_fwd_1d"] = spy_ret.shift(-1)
        feat_df["target_spy_fwd_5d"] = (1.0 + spy_ret.shift(-5)).rolling(5).apply(np.prod, raw=True) - 1.0
        feat_df["target_trade_win"] = (feat_df["target_spy_fwd_1d"] > 0).astype(int)

        # Zero out leading uninitialized periods without using future data (strictly avoiding lookahead)
        feat_df = feat_df.fillna(0.0)

        # Save processed features
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        feat_df.to_csv(PROCESSED_DATA_DIR / "engineered_features.csv")
        logger.info(f"Engineered {len(feat_df.columns)} quantitative features across {len(feat_df)} trading days.")
        
        return feat_df

if __name__ == "__main__":
    pass
