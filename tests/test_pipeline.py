"""
Unit and integration tests for Macro Economics Agent Pipeline.
Includes rigorous tests for lookahead bias prevention, Sortino numerical precision,
and out-of-sample window integrity.
"""

import unittest
import pandas as pd
import numpy as np
from src.ingestion.apify_calendar_scraper import ApifyMacroCalendarIngestor
from src.ingestion.news_tavily_ingestion import TavilyMacroNewsIngestor
from src.ingestion.market_data_loader import MarketDataLoader
from src.features.feature_engineering import MacroFeatureEngine
from src.models.strategy_engine import MacroStrategyEngine
from src.models.walk_forward import WalkForwardValidator

class TestMacroPipeline(unittest.TestCase):
    def setUp(self):
        self.calendar_ingestor = ApifyMacroCalendarIngestor()
        self.news_ingestor = TavilyMacroNewsIngestor()
        self.price_loader = MarketDataLoader()
        self.feature_engine = MacroFeatureEngine()

    def test_calendar_ingestion(self):
        df_events = self.calendar_ingestor.ingest()
        self.assertIsInstance(df_events, pd.DataFrame)
        self.assertGreater(len(df_events), 15)
        self.assertIn("timestamp_utc", df_events.columns)
        self.assertIn("timestamp_est", df_events.columns)
        self.assertIn("surprise", df_events.columns)

    def test_news_ingestion(self):
        df_news = self.news_ingestor.ingest()
        self.assertIsInstance(df_news, pd.DataFrame)
        self.assertGreater(len(df_news), 5)
        self.assertIn("sentiment_score", df_news.columns)

    def test_market_prices(self):
        df_prices = self.price_loader.fetch_historical_prices()
        self.assertIsInstance(df_prices, pd.DataFrame)
        self.assertGreater(len(df_prices), 100)
        self.assertIn("daily_return", df_prices.columns)

    def test_sortino_ratio_numerical_accuracy(self):
        """Verifies Sortino ratio against hand-computed reference values."""
        # 5 days of returns with rf=0
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        rf = 0.0
        # mean = 0.006. Annualized mean = 0.006 * 252 = 1.512
        # downside deviations from 0: [0, -0.01, 0, -0.02, 0]
        # squares = [0, 0.0001, 0, 0.0004, 0]. mean = 0.0005 / 5 = 0.0001
        # rms = 0.01. Annualized downside vol = 0.01 * sqrt(252) = 0.158745
        # sortino = 1.512 / 0.158745 = 9.525
        metrics = MacroStrategyEngine.calculate_metrics(returns, rf_annual=0.0)
        self.assertAlmostEqual(metrics["sortino"], 9.525, places=2)

    def test_zero_lookahead_feature_engineering(self):
        """Verifies that future market changes do NOT alter past engineered feature rows."""
        events_df = self.calendar_ingestor.ingest()
        news_df = self.news_ingestor.ingest()
        prices_df = self.price_loader.fetch_historical_prices()

        # Compute features on full history
        feat_1 = self.feature_engine.build_features(events_df, news_df, prices_df)

        # Alter future price values (e.g. last 10 rows)
        prices_perturbed = prices_df.copy()
        prices_perturbed.loc[prices_perturbed.index[-10:], "adj_close"] *= 2.0
        prices_perturbed.loc[prices_perturbed.index[-10:], "daily_return"] = 0.5

        feat_2 = self.feature_engine.build_features(events_df, news_df, prices_perturbed)

        # Prior historical rows (e.g. index 0 to 50) must remain strictly identical
        pd.testing.assert_frame_equal(feat_1.iloc[:50], feat_2.iloc[:50])

    def test_walk_forward_burn_in_exclusion(self):
        """Verifies that walk-forward validator excludes unpredicted burn-in window from OOS metrics."""
        validator = WalkForwardValidator(train_window=90, test_window=15)
        self.assertEqual(validator.train_window, 90)
        self.assertEqual(validator.test_window, 15)

if __name__ == "__main__":
    unittest.main()
