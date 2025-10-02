"""
Feast Entity Definitions
Entities represent the primary keys for feature retrieval
"""
from feast import Entity
from feast.value_type import ValueType

# Symbol entity - for equity-level features
symbol_entity = Entity(
    name="symbol",
    value_type=ValueType.STRING,
    description="Stock ticker symbol (e.g., AAPL, MSFT)",
    join_keys=["symbol"]
)

# Portfolio entity - for portfolio-level features
portfolio_entity = Entity(
    name="portfolio_id",
    value_type=ValueType.STRING,
    description="Portfolio identifier",
    join_keys=["portfolio_id"]
)

# Sector entity - for sector/industry features
sector_entity = Entity(
    name="sector",
    value_type=ValueType.STRING,
    description="GICS sector classification",
    join_keys=["sector"]
)
