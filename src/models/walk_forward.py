"""
Walk-Forward Cross-Validation and Machine Learning Predictive Model.
Implements out-of-sample rolling train/test splits, prevents lookahead bias,
and trains ensemble predictors for forward trade P&L and win rate.
"""

import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from config.settings import WALK_FORWARD_TRAIN_DAYS, WALK_FORWARD_TEST_DAYS, RANDOM_STATE
from src.models.strategy_engine import MacroStrategyEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class WalkForwardValidator:
    def __init__(
        self,
        train_window: int = WALK_FORWARD_TRAIN_DAYS,
        test_window: int = WALK_FORWARD_TEST_DAYS
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.feature_cols = [
            "days_since_fomc", "days_until_fomc", "days_since_cpi", "days_since_nfp",
            "fomc_decay", "cpi_decay", "nfp_decay", "is_pre_fomc", "is_post_fomc",
            "latest_cpi_surprise", "latest_nfp_surprise", "current_fed_rate",
            "macro_shock_index", "raw_sentiment", "rolling_macro_sentiment",
            "spy_ret_5d", "tlt_ret_5d", "gld_ret_5d", "uup_ret_5d",
            "spy_vol_20d", "tlt_vol_20d", "yield_curve_slope"
        ]

    def run_walk_forward(self, df_features: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, Dict[str, float]]:
        """
        Executes rolling walk-forward out-of-sample validation.
        Returns:
            - out_of_sample_signals: pd.Series of model signals
            - strategy_performance_df: Daily P&L series across strategies
            - metrics_summary: Dict of institutional risk/return metrics
        """
        logger.info(f"Initiating Walk-Forward Validation (Train Window: {self.train_window}d, Test Window: {self.test_window}d)...")
        
        # Ensure only available feature columns are selected
        avail_features = [c for c in self.feature_cols if c in df_features.columns]
        X = df_features[avail_features].copy()
        y_ret = df_features["target_spy_fwd_1d"].copy()
        y_win = df_features["target_trade_win"].copy()

        n_samples = len(df_features)
        oos_predictions = pd.Series(index=df_features.index, dtype=float)
        feature_importances = []

        # Iterate in rolling windows
        start_idx = self.train_window
        total_folds = 0

        while start_idx < n_samples:
            end_test_idx = min(start_idx + self.test_window, n_samples)
            
            # Strict historical train split (no future data leakage)
            train_idx_start = max(0, start_idx - self.train_window)
            X_train = X.iloc[train_idx_start:start_idx]
            y_train_ret = y_ret.iloc[train_idx_start:start_idx]
            y_train_win = y_win.iloc[train_idx_start:start_idx]

            X_test = X.iloc[start_idx:end_test_idx]

            # Standardize features using training statistics only
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Fit Ensemble Regressor for forward return
            regressor = RandomForestRegressor(
                n_estimators=100,
                max_depth=4,
                min_samples_split=4,
                random_state=RANDOM_STATE
            )
            regressor.fit(X_train_scaled, y_train_ret)
            pred_fwd_ret = regressor.predict(X_test_scaled)

            # Fit Classifier for Win Probability
            classifier = GradientBoostingClassifier(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.05,
                random_state=RANDOM_STATE
            )
            classifier.fit(X_train_scaled, y_train_win)
            pred_win_prob = classifier.predict_proba(X_test_scaled)[:, 1]

            # Combined Signal: Directional magnitude modulated by win probability confidence
            # Center win probability around 0.5 (-0.5 to +0.5)
            confidence = (pred_win_prob - 0.5) * 2.0
            fold_signal = np.sign(pred_fwd_ret) * (0.5 * np.abs(pred_fwd_ret) / 0.01 + 0.5 * confidence)

            oos_predictions.iloc[start_idx:end_test_idx] = fold_signal
            feature_importances.append(regressor.feature_importances_)
            
            total_folds += 1
            start_idx = end_test_idx

        # Neutral zero exposure during burn-in (strictly avoiding lookahead backfill)
        oos_predictions.iloc[:self.train_window] = 0.0
        logger.info(f"Completed {total_folds} Walk-Forward Out-of-Sample folds ({n_samples - self.train_window} pure OOS days).")

        # Compute average feature importance
        mean_importance = np.mean(feature_importances, axis=0) if len(feature_importances) > 0 else np.zeros(len(avail_features))
        importance_df = pd.DataFrame({
            "feature": avail_features,
            "importance": mean_importance
        }).sort_values(by="importance", ascending=False)

        # Run Backtest with Strategy Engine
        strategy_engine = MacroStrategyEngine()
        strategy_pnl_df = strategy_engine.backtest_strategies(df_features, oos_predictions)

        # Compute Performance Metrics strictly across the pure Out-of-Sample evaluation window
        oos_pnl_df = strategy_pnl_df.iloc[self.train_window:]
        metrics_summary = {}
        for col in oos_pnl_df.columns:
            m = strategy_engine.calculate_metrics(oos_pnl_df[col])
            metrics_summary[col] = m

        return oos_predictions, strategy_pnl_df, metrics_summary, importance_df
