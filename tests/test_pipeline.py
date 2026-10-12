"""
Unit and integration tests for Macro Economics Agent Pipeline.
"""

import unittest
import pandas as pd
import numpy as np
from src.ingestion.apify_calendar_scraper import ApifyMacroCalendarIngestor
from src.ingestion.news_tavily_ingestion import TavilyMacroNewsIngestor
from src.ingestion.market_data_loader import MarketDataLoader
from src.features.feature_engineering import MacroFeatureEngine
from src.models.strategy_engine import MacroStrategyEngine

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

    def test_strategy_metric_calculations(self):
        # Synthetic returns series with known positive mean
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))
        metrics = MacroStrategyEngine.calculate_metrics(returns)
        self.assertIn("sharpe", metrics)
        self.assertIn("sortino", metrics)
        self.assertIn("max_drawdown_pct", metrics)
        self.assertGreater(metrics["max_drawdown_pct"], 0.0)

if __name__ == "__main__":
    unittest.main()
