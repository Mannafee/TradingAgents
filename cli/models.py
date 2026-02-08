from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel


class AnalystType(str, Enum):
    MARKET = "market"
    SOCIAL = "social"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"


class RiskToleranceChoice(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class StockUniverseChoice(str, Enum):
    SP500_TOP50 = "sp500_top50"
    NASDAQ_TOP30 = "nasdaq_top30"
    ETF_POPULAR = "etf_popular"
    CUSTOM = "custom"
