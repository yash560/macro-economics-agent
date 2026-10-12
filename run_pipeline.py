"""
Master Execution Pipeline for Macro Economics Agent.
Runs complete end-to-end quantitative workflow:
1. Data Ingestion (Apify Calendar, Tavily News, Multi-Asset Market Data)
2. Database Persistence (SQLite)
3. Quantitative Feature Engineering (Macro surprises, event lags, regime proxies)
4. Walk-Forward Cross-Validation & ML Ensemble Predictor
5. Strategy Permutation Backtesting with Transaction Costs
6. Visual Matrix Generation & Publication Output
7. Automated Metric Verification
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH, OUTPUTS_DIR, REPORTS_DIR
from src.database.db_manager import DatabaseManager
from src.ingestion.apify_calendar_scraper import ApifyMacroCalendarIngestor
from src.ingestion.news_tavily_ingestion import TavilyMacroNewsIngestor
from src.ingestion.market_data_loader import MarketDataLoader
from src.features.feature_engineering import MacroFeatureEngine
from src.models.walk_forward import WalkForwardValidator
from src.visualization.matrix_generator import VisualMatrixGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run():
    logger.info("=" * 70)
    logger.info("STARTING MACRO ECONOMICS AGENT PIPELINE EXECUTION")
    logger.info("=" * 70)

    # Step 1: Database Initialization
    logger.info("[Step 1/6] Initializing SQLite Relational Database...")
    db = DatabaseManager(DB_PATH)

    # Step 2: Data Ingestion
    logger.info("[Step 2/6] Ingesting Macro Calendar, News & Market Prices...")
    calendar_ingestor = ApifyMacroCalendarIngestor()
    df_events = calendar_ingestor.ingest()
    db.insert_events(df_events)

    news_ingestor = TavilyMacroNewsIngestor()
    df_news = news_ingestor.ingest()
    db.insert_news(df_news)

    price_loader = MarketDataLoader()
    df_prices = price_loader.fetch_historical_prices()
    db.insert_prices(df_prices)

    # Step 3: Feature Engineering
    logger.info("[Step 3/6] Engineering Time-Based & Macroeconomic Features...")
    feature_engine = MacroFeatureEngine()
    df_features = feature_engine.build_features(df_events, df_news, df_prices)
    db.insert_features(df_features)

    # Step 4: Machine Learning & Walk-Forward Validation
    logger.info("[Step 4/6] Training Ensemble ML Model via Walk-Forward Out-Of-Sample Validation...")
    validator = WalkForwardValidator()
    oos_signals, strategy_pnl_df, metrics_summary, importance_df = validator.run_walk_forward(df_features)

    # Step 5: Visual Matrix Outputs
    logger.info("[Step 5/6] Generating Visual Matrix Outputs in outputs/...")
    matrix_gen = VisualMatrixGenerator(OUTPUTS_DIR)
    matrix_gen.generate_all_visuals(df_features, strategy_pnl_df, metrics_summary, importance_df)

    # Step 6: Summary Reporting
    logger.info("[Step 6/6] Execution Complete. Summary Metrics:")
    logger.info("-" * 70)
    for strat, m in metrics_summary.items():
        logger.info(f"Strategy: {strat:<25} | Sharpe: {m['sharpe']:>5.2f} | Sortino: {m['sortino']:>5.2f} | MaxDD: {m['max_drawdown_pct']:>5.2f}% | WinRate: {m['win_rate_pct']:>5.2f}% | Return: {m['total_return_pct']:>5.2f}%")
    logger.info("-" * 70)

    return df_features, strategy_pnl_df, metrics_summary, importance_df

if __name__ == "__main__":
    run()
