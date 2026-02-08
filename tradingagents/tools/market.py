"""Market data tool schemas and executors."""

from tradingagents.data.market import get_stock_data as _get_stock_data
from tradingagents.data.market import get_indicators as _get_indicators

GET_STOCK_DATA_SCHEMA = {
    "type": "function",
    "name": "get_stock_data",
    "description": (
        "Retrieve stock price data (OHLCV) for a given ticker symbol. "
        "Returns a formatted CSV with Open, High, Low, Close, Volume columns."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Ticker symbol of the company, e.g. AAPL, TSM",
            },
            "start_date": {
                "type": "string",
                "description": "Start date in yyyy-mm-dd format",
            },
            "end_date": {
                "type": "string",
                "description": "End date in yyyy-mm-dd format",
            },
        },
        "required": ["symbol", "start_date", "end_date"],
        "additionalProperties": False,
    },
    "strict": True,
}

GET_INDICATORS_SCHEMA = {
    "type": "function",
    "name": "get_indicators",
    "description": (
        "Retrieve technical indicators for a given ticker symbol. "
        "Supported indicators: close_50_sma, close_200_sma, close_10_ema, "
        "macd, macds, macdh, rsi, boll, boll_ub, boll_lb, atr, vwma, mfi."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Ticker symbol of the company, e.g. AAPL, TSM",
            },
            "indicator": {
                "type": "string",
                "description": "Technical indicator name (e.g. rsi, macd, boll)",
            },
            "curr_date": {
                "type": "string",
                "description": "The current trading date, YYYY-mm-dd",
            },
            "look_back_days": {
                "type": "integer",
                "description": "How many days to look back (default 30)",
            },
        },
        "required": ["symbol", "indicator", "curr_date"],
        "additionalProperties": False,
    },
    "strict": False,
}

SCHEMAS = [GET_STOCK_DATA_SCHEMA, GET_INDICATORS_SCHEMA]

EXECUTORS = {
    "get_stock_data": _get_stock_data,
    "get_indicators": _get_indicators,
}
