#!/usr/bin/env python3
"""
Initial Data Population Script
Populates first-print fundamentals, universe tables, and initial historical data
"""
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def populate_seed_data(conn):
    """Populate basic seed data (users, portfolios, watchlists)"""
    logger.info("Populating seed data...")

    seed_file = project_root / 'migrations' / 'seed_data.sql'
    if not seed_file.exists():
        logger.warning(f"Seed file not found: {seed_file}")
        return

    with open(seed_file, 'r', encoding='utf-8') as f:
        sql = f.read()

    try:
        await conn.execute(sql)
        logger.info("✓ Seed data populated successfully")
    except Exception as e:
        logger.error(f"Error populating seed data: {e}")
        raise


async def initialize_universe_tables(conn):
    """Initialize universe tracking tables"""
    logger.info("Initializing universe tables...")

    # Import the universe loader
    sys.path.insert(0, str(project_root / 'universe'))
    from ptfs_loader import PortfolioUniverseLoader

    loader = PortfolioUniverseLoader(database_url=None)
    loader.database_url = None  # Will use connection we provide

    # Create the tables
    await loader.initialize_universe_tables()
    logger.info("✓ Universe tables initialized")


async def populate_sp500_constituents(conn):
    """Populate S&P 500 constituents"""
    logger.info("Populating S&P 500 constituents...")

    # Sample S&P 500 symbols - in production this would come from a data provider
    sp500_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ',
        'V', 'WMT', 'JPM', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'LLY',
        'ABBV', 'KO', 'PEP', 'AVGO', 'COST', 'ADBE', 'MCD', 'TMO', 'CSCO', 'ACN',
        'ABT', 'NFLX', 'NKE', 'DHR', 'TXN', 'DIS', 'VZ', 'PM', 'INTC', 'WFC',
        'CRM', 'NEE', 'CMCSA', 'AMD', 'UPS', 'RTX', 'HON', 'ORCL', 'QCOM', 'INTU'
    ]

    effective_date = datetime(2024, 1, 1).date()

    insert_query = """
        INSERT INTO universe_constituents
        (symbol, universe_type, effective_date, listing_status, data_source)
        VALUES ($1, 'sp500', $2, 'active', 'manual')
        ON CONFLICT (symbol, universe_type, effective_date) DO NOTHING
    """

    count = 0
    for symbol in sp500_symbols:
        try:
            await conn.execute(insert_query, symbol, effective_date)
            count += 1
        except Exception as e:
            logger.error(f"Error inserting {symbol}: {e}")

    logger.info(f"✓ Populated {count} S&P 500 constituents")


async def populate_sample_fundamentals(conn):
    """Populate sample first-print fundamentals data"""
    logger.info("Populating sample fundamentals data...")

    # Sample fundamentals for a few symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    # Insert into fundamentals_first_print table
    insert_query = """
        INSERT INTO fundamentals_first_print
        (symbol, report_date, period_end_date, fiscal_quarter, fiscal_year,
         revenue, net_income, earnings_per_share, total_assets, total_debt, total_equity,
         operating_cash_flow, free_cash_flow, data_source, first_print_timestamp)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        ON CONFLICT (symbol, report_date, period_end_date) DO NOTHING
    """

    count = 0
    base_date = datetime(2024, 1, 1)

    for i, symbol in enumerate(symbols):
        # Create sample data for last 4 quarters
        for q in range(4):
            quarter_date = base_date - timedelta(days=q*90)
            filing_date = quarter_date + timedelta(days=45)  # Filed 45 days after quarter end

            # Sample values (not real data)
            revenue = 100_000_000_000 + (i * 10_000_000_000) + (q * 5_000_000_000)
            net_income = revenue * 0.25
            eps = 1.50 + (i * 0.5) + (q * 0.1)

            try:
                await conn.execute(
                    insert_query,
                    symbol,
                    filing_date.date(),
                    quarter_date.date(),
                    (q % 4) + 1,  # Fiscal quarter 1-4
                    2024 if q < 4 else 2023,  # Fiscal year
                    float(revenue),
                    float(net_income),
                    float(eps),
                    float(revenue * 2),  # Total assets
                    float(revenue * 0.5),  # Total liabilities
                    float(revenue * 1.5),  # Shareholders equity
                    float(net_income * 1.2),  # Operating cash flow
                    float(net_income * 1.0),  # Free cash flow
                    'sample_data',  # Data source
                    filing_date  # First reported at
                )
                count += 1
            except Exception as e:
                logger.error(f"Error inserting fundamentals for {symbol} Q{q}: {e}")

    logger.info(f"✓ Populated {count} fundamentals records")


async def populate_sample_market_data(conn):
    """Populate sample market data for testing"""
    logger.info("Populating sample market data...")

    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ']

    # Insert daily candles for the past 30 days
    insert_query = """
        INSERT INTO candles (symbol, ts, open, high, low, close, volume)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (symbol, ts) DO NOTHING
    """

    count = 0
    base_date = datetime(2024, 10, 1, 9, 30, 0)  # Market open time

    for symbol in symbols:
        base_price = 100.0 + (len(symbol) * 10)  # Simple price based on symbol

        for day in range(30):
            # Generate daily data
            date = base_date - timedelta(days=day)

            # Skip weekends
            if date.weekday() >= 5:
                continue

            # Simple random-walk price simulation
            daily_change = (day % 3 - 1) * 0.5  # -0.5, 0, or 0.5

            open_price = base_price + daily_change
            close_price = open_price + (day % 2 - 0.5)
            high_price = max(open_price, close_price) + 0.5
            low_price = min(open_price, close_price) - 0.5
            volume = 10_000_000 + (day * 100_000)

            try:
                await conn.execute(
                    insert_query,
                    symbol,
                    date,
                    float(open_price),
                    float(high_price),
                    float(low_price),
                    float(close_price),
                    int(volume)
                )
                count += 1
            except Exception as e:
                logger.error(f"Error inserting market data for {symbol} on {date}: {e}")

    logger.info(f"✓ Populated {count} market data records")


async def main():
    """Main data population function"""
    print("=" * 60)
    print("INITIAL DATA POPULATION")
    print("=" * 60)
    print()

    # Get database connection
    database_url = os.getenv('DATABASE_URL',
        'postgresql://trading_user:trading_pass@localhost:5432/trading_db')

    try:
        import asyncpg
        conn = await asyncpg.connect(database_url)
        logger.info("✓ Connected to database")
    except Exception as e:
        logger.error(f"✗ Failed to connect to database: {e}")
        return 1

    try:
        # Run all population tasks
        await populate_seed_data(conn)
        await initialize_universe_tables(conn)
        await populate_sp500_constituents(conn)
        await populate_sample_fundamentals(conn)
        await populate_sample_market_data(conn)

        print()
        print("=" * 60)
        print("DATA POPULATION COMPLETE")
        print("=" * 60)
        print()
        print("Summary:")
        print("  ✓ Seed data (users, portfolios, watchlists)")
        print("  ✓ Universe tables initialized")
        print("  ✓ S&P 500 constituents")
        print("  ✓ Sample fundamentals (first-print)")
        print("  ✓ Sample market data (30 days)")
        print()
        print("Next steps:")
        print("  1. Verify data: SELECT COUNT(*) FROM universe_constituents;")
        print("  2. Verify fundamentals: SELECT COUNT(*) FROM fundamentals_first_print;")
        print("  3. Verify market data: SELECT COUNT(*) FROM candles;")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Error during data population: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        await conn.close()


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
