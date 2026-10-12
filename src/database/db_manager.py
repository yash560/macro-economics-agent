"""
Database Manager for Macro Economics Agent.
Manages SQLite storage for macroeconomic events, financial news, market prices, and features.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from config.settings import DB_PATH

class DatabaseManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initializes database schema with proper indexes."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Macroeconomic Events Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS macro_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                timestamp_est TEXT NOT NULL,
                date TEXT NOT NULL,
                actual REAL,
                forecast REAL,
                previous REAL,
                surprise REAL,
                surprise_std REAL,
                impact_level TEXT,
                source TEXT DEFAULT 'apify_calendar'
            );
            """)

            # 2. Economic News Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS macro_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                date TEXT NOT NULL,
                headline TEXT NOT NULL,
                source TEXT,
                sentiment_score REAL,
                relevance_score REAL,
                summary TEXT
            );
            """)

            # 3. Market Prices Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_prices (
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                daily_return REAL,
                PRIMARY KEY (date, symbol)
            );
            """)

            # 4. Master Feature Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS master_features (
                date TEXT PRIMARY KEY,
                cpi_surprise REAL,
                nfp_surprise REAL,
                fomc_decision_rate REAL,
                rolling_macro_sentiment REAL,
                days_since_fomc INTEGER,
                days_until_fomc INTEGER,
                spy_ret_5d REAL,
                tlt_ret_5d REAL,
                spy_vol_20d REAL,
                tlt_vol_20d REAL,
                yield_curve_slope REAL
            );
            """)

            # Create performance indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_date ON macro_events(date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_date ON macro_news(date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_date_sym ON market_prices(date, symbol);")
            conn.commit()

    def insert_events(self, df: pd.DataFrame):
        with self.get_connection() as conn:
            df.to_sql("macro_events", conn, if_exists="append", index=False)

    def insert_news(self, df: pd.DataFrame):
        with self.get_connection() as conn:
            df.to_sql("macro_news", conn, if_exists="append", index=False)

    def insert_prices(self, df: pd.DataFrame):
        with self.get_connection() as conn:
            df.to_sql("market_prices", conn, if_exists="replace", index=False)

    def insert_features(self, df: pd.DataFrame):
        with self.get_connection() as conn:
            df.to_sql("master_features", conn, if_exists="replace", index=False)

    def query(self, sql: str, params=()) -> pd.DataFrame:
        with self.get_connection() as conn:
            return pd.read_sql_query(sql, conn, params=params)
