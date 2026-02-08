"""News data tool schemas and executors."""

from tradingagents.data.news import get_news as _get_news
from tradingagents.data.news import get_global_news as _get_global_news
from tradingagents.data.fundamentals import get_insider_transactions as _get_insider_transactions

GET_NEWS_SCHEMA = {
    "type": "function",
    "name": "get_news",
    "description": "Retrieve news data for a given ticker symbol.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol"},
            "start_date": {"type": "string", "description": "Start date in yyyy-mm-dd format"},
            "end_date": {"type": "string", "description": "End date in yyyy-mm-dd format"},
        },
        "required": ["ticker", "start_date", "end_date"],
        "additionalProperties": False,
    },
    "strict": True,
}

GET_GLOBAL_NEWS_SCHEMA = {
    "type": "function",
    "name": "get_global_news",
    "description": "Retrieve global/macroeconomic news.",
    "parameters": {
        "type": "object",
        "properties": {
            "curr_date": {"type": "string", "description": "Current date in yyyy-mm-dd format"},
            "look_back_days": {"type": "integer", "description": "Number of days to look back (default 7)"},
            "limit": {"type": "integer", "description": "Maximum number of articles (default 5)"},
        },
        "required": ["curr_date"],
        "additionalProperties": False,
    },
    "strict": False,
}

GET_INSIDER_TRANSACTIONS_SCHEMA = {
    "type": "function",
    "name": "get_insider_transactions",
    "description": "Retrieve insider transaction information about a company.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol"},
        },
        "required": ["ticker"],
        "additionalProperties": False,
    },
    "strict": True,
}

SCHEMAS = [GET_NEWS_SCHEMA, GET_GLOBAL_NEWS_SCHEMA, GET_INSIDER_TRANSACTIONS_SCHEMA]

EXECUTORS = {
    "get_news": _get_news,
    "get_global_news": _get_global_news,
    "get_insider_transactions": _get_insider_transactions,
}
