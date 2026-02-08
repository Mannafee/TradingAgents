"""Fundamentals data tool schemas and executors."""

from tradingagents.data.fundamentals import (
    get_fundamentals as _get_fundamentals,
    get_balance_sheet as _get_balance_sheet,
    get_cashflow as _get_cashflow,
    get_income_statement as _get_income_statement,
)

GET_FUNDAMENTALS_SCHEMA = {
    "type": "function",
    "name": "get_fundamentals",
    "description": "Retrieve comprehensive fundamental data for a given ticker symbol.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol"},
            "curr_date": {"type": "string", "description": "Current date, yyyy-mm-dd"},
        },
        "required": ["ticker", "curr_date"],
        "additionalProperties": False,
    },
    "strict": True,
}

GET_BALANCE_SHEET_SCHEMA = {
    "type": "function",
    "name": "get_balance_sheet",
    "description": "Retrieve balance sheet data for a given ticker symbol.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol"},
            "freq": {
                "type": "string",
                "description": "Reporting frequency: annual or quarterly (default quarterly)",
            },
            "curr_date": {"type": "string", "description": "Current date, yyyy-mm-dd"},
        },
        "required": ["ticker"],
        "additionalProperties": False,
    },
    "strict": False,
}

GET_CASHFLOW_SCHEMA = {
    "type": "function",
    "name": "get_cashflow",
    "description": "Retrieve cash flow statement data for a given ticker symbol.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol"},
            "freq": {
                "type": "string",
                "description": "Reporting frequency: annual or quarterly (default quarterly)",
            },
            "curr_date": {"type": "string", "description": "Current date, yyyy-mm-dd"},
        },
        "required": ["ticker"],
        "additionalProperties": False,
    },
    "strict": False,
}

GET_INCOME_STATEMENT_SCHEMA = {
    "type": "function",
    "name": "get_income_statement",
    "description": "Retrieve income statement data for a given ticker symbol.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol"},
            "freq": {
                "type": "string",
                "description": "Reporting frequency: annual or quarterly (default quarterly)",
            },
            "curr_date": {"type": "string", "description": "Current date, yyyy-mm-dd"},
        },
        "required": ["ticker"],
        "additionalProperties": False,
    },
    "strict": False,
}

SCHEMAS = [
    GET_FUNDAMENTALS_SCHEMA,
    GET_BALANCE_SHEET_SCHEMA,
    GET_CASHFLOW_SCHEMA,
    GET_INCOME_STATEMENT_SCHEMA,
]

EXECUTORS = {
    "get_fundamentals": _get_fundamentals,
    "get_balance_sheet": _get_balance_sheet,
    "get_cashflow": _get_cashflow,
    "get_income_statement": _get_income_statement,
}
