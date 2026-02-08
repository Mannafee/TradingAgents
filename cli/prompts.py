"""Interactive prompts for TradingAgents CLI (Codex-only)."""

import questionary
from typing import List, Tuple
from rich.console import Console

from cli.models import AnalystType

console = Console()

ANALYST_ORDER = [
    ("Market Analyst", AnalystType.MARKET),
    ("Social Media Analyst", AnalystType.SOCIAL),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
]


def get_ticker() -> str:
    """Prompt the user to enter a ticker symbol."""
    ticker = questionary.text(
        "Enter the ticker symbol to analyze:",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a valid ticker symbol.",
        style=questionary.Style([
            ("text", "fg:green"),
            ("highlighted", "noinherit"),
        ]),
    ).ask()

    if not ticker:
        console.print("\n[red]No ticker symbol provided. Exiting...[/red]")
        exit(1)

    return ticker.strip().upper()


def get_analysis_date() -> str:
    """Prompt the user to enter a date in YYYY-MM-DD format."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        "Enter the analysis date (YYYY-MM-DD):",
        validate=lambda x: validate_date(x.strip())
        or "Please enter a valid date in YYYY-MM-DD format.",
        style=questionary.Style([
            ("text", "fg:green"),
            ("highlighted", "noinherit"),
        ]),
    ).ask()

    if not date:
        console.print("\n[red]No date provided. Exiting...[/red]")
        exit(1)

    return date.strip()


def select_analysts() -> List[AnalystType]:
    """Select analysts using an interactive checkbox."""
    choices = questionary.checkbox(
        "Select Your [Analysts Team]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in ANALYST_ORDER
        ],
        instruction="\n- Press Space to select/unselect analysts\n- Press 'a' to select/unselect all\n- Press Enter when done",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style([
            ("checkbox-selected", "fg:green"),
            ("selected", "fg:green noinherit"),
            ("highlighted", "noinherit"),
            ("pointer", "noinherit"),
        ]),
    ).ask()

    if not choices:
        console.print("\n[red]No analysts selected. Exiting...[/red]")
        exit(1)

    return choices


def select_research_depth() -> int:
    """Select research depth using an interactive selection."""
    DEPTH_OPTIONS = [
        ("Shallow - Quick research, few debate and strategy discussion rounds", 1),
        ("Medium - Middle ground, moderate debate rounds and strategy discussion", 3),
        ("Deep - Comprehensive research, in depth debate and strategy discussion", 5),
    ]

    choice = questionary.select(
        "Select Your [Research Depth]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in DEPTH_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style([
            ("selected", "fg:yellow noinherit"),
            ("highlighted", "fg:yellow noinherit"),
            ("pointer", "fg:yellow noinherit"),
        ]),
    ).ask()

    if choice is None:
        console.print("\n[red]No research depth selected. Exiting...[/red]")
        exit(1)

    return choice


def select_quick_model() -> str:
    """Select the quick-thinking (shallow) model for Codex."""
    OPTIONS = [
        ("GPT-5.1 Codex Mini - Cheaper, faster", "gpt-5.1-codex-mini"),
        ("GPT-5.2 - Frontier model, knowledge + reasoning", "gpt-5.2"),
        ("GPT-5.2 Codex - Frontier agentic coding", "gpt-5.2-codex"),
        ("GPT-5.3 Codex - Latest agentic coding", "gpt-5.3-codex"),
    ]

    choice = questionary.select(
        "Select Your [Quick-Thinking LLM]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style([
            ("selected", "fg:magenta noinherit"),
            ("highlighted", "fg:magenta noinherit"),
            ("pointer", "fg:magenta noinherit"),
        ]),
    ).ask()

    if choice is None:
        console.print("\n[red]No model selected. Exiting...[/red]")
        exit(1)

    return choice


def select_deep_model() -> str:
    """Select the deep-thinking model for Codex."""
    OPTIONS = [
        ("GPT-5.2 Codex - Frontier agentic coding (default)", "gpt-5.2-codex"),
        ("GPT-5.3 Codex - Latest agentic coding", "gpt-5.3-codex"),
        ("GPT-5.1 Codex Max - Deep and fast reasoning", "gpt-5.1-codex-max"),
        ("GPT-5.2 - Frontier model, knowledge + reasoning", "gpt-5.2"),
        ("GPT-5.1 Codex Mini - Cheaper, faster", "gpt-5.1-codex-mini"),
    ]

    choice = questionary.select(
        "Select Your [Deep-Thinking LLM]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style([
            ("selected", "fg:magenta noinherit"),
            ("highlighted", "fg:magenta noinherit"),
            ("pointer", "fg:magenta noinherit"),
        ]),
    ).ask()

    if choice is None:
        console.print("\n[red]No model selected. Exiting...[/red]")
        exit(1)

    return choice


def run_codex_login():
    """Run the Codex OAuth login flow and return a CodexClient.

    Returns the configured CodexClient.
    """
    from tradingagents.client.auth import CodexOAuth
    from tradingagents.client.codex import CodexClient

    console.print("\n[cyan]Opening browser for ChatGPT login...[/cyan]")
    console.print("[dim]If the browser doesn't open, check the terminal for the URL.[/dim]")

    auth = CodexOAuth()
    result = auth.load_or_login()

    import os
    os.environ["OPENAI_API_KEY"] = result.access_token
    if result.account_id:
        os.environ["CHATGPT_ACCOUNT_ID"] = result.account_id
    os.environ["CODEX_OAUTH_MODE"] = "1"

    console.print("[green]Login successful![/green]")

    return CodexClient(
        access_token=result.access_token,
        base_url=result.base_url,
    )


# --- Portfolio prompts ---

def get_budget() -> Tuple[float, str]:
    """Prompt for budget amount and currency."""
    CURRENCY_OPTIONS = [
        ("USD - US Dollar", "USD"),
        ("EUR - Euro", "EUR"),
        ("GBP - British Pound", "GBP"),
        ("JPY - Japanese Yen", "JPY"),
        ("CHF - Swiss Franc", "CHF"),
        ("CAD - Canadian Dollar", "CAD"),
        ("AUD - Australian Dollar", "AUD"),
    ]

    currency_choice = questionary.select(
        "Select your currency:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in CURRENCY_OPTIONS
        ],
        style=questionary.Style([
            ("selected", "fg:green noinherit"),
            ("highlighted", "fg:green noinherit"),
            ("pointer", "fg:green noinherit"),
        ]),
    ).ask()

    if not currency_choice:
        console.print("\n[red]No currency selected. Exiting...[/red]")
        exit(1)

    amount_str = questionary.text(
        f"Enter your investment budget ({currency_choice}):",
        validate=lambda x: (
            x.strip().replace(".", "", 1).isdigit() and float(x.strip()) > 0
        ) or "Please enter a positive number.",
        style=questionary.Style([
            ("text", "fg:green"),
            ("highlighted", "noinherit"),
        ]),
    ).ask()

    if not amount_str:
        console.print("\n[red]No budget provided. Exiting...[/red]")
        exit(1)

    return float(amount_str.strip()), currency_choice


def get_time_horizon() -> int:
    """Prompt for time horizon in days."""
    TIME_OPTIONS = [
        ("1 day (intraday/swing)", 1),
        ("1 week", 7),
        ("2 weeks", 14),
        ("1 month", 30),
        ("3 months", 90),
    ]

    choice = questionary.select(
        "Select your investment time horizon:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in TIME_OPTIONS
        ],
        style=questionary.Style([
            ("selected", "fg:yellow noinherit"),
            ("highlighted", "fg:yellow noinherit"),
            ("pointer", "fg:yellow noinherit"),
        ]),
    ).ask()

    if choice is None:
        console.print("\n[red]No time horizon selected. Exiting...[/red]")
        exit(1)

    return choice


def select_risk_tolerance() -> str:
    """Prompt for risk tolerance level."""
    RISK_OPTIONS = [
        ("Conservative - Prioritize capital preservation, lower risk", "conservative"),
        ("Moderate - Balanced risk and reward", "moderate"),
        ("Aggressive - Maximize potential returns, higher risk", "aggressive"),
    ]

    choice = questionary.select(
        "Select your risk tolerance:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in RISK_OPTIONS
        ],
        style=questionary.Style([
            ("selected", "fg:yellow noinherit"),
            ("highlighted", "fg:yellow noinherit"),
            ("pointer", "fg:yellow noinherit"),
        ]),
    ).ask()

    if choice is None:
        console.print("\n[red]No risk tolerance selected. Exiting...[/red]")
        exit(1)

    return choice


def select_stock_universe() -> str:
    """Prompt for stock universe to screen."""
    UNIVERSE_OPTIONS = [
        ("S&P 500 Top 50 - Largest, most liquid US stocks (recommended)", "sp500_top50"),
        ("NASDAQ Top 30 - Tech-focused growth stocks", "nasdaq_top30"),
        ("Popular ETFs - Diversified index & sector ETFs", "etf_popular"),
    ]

    choice = questionary.select(
        "Select stock universe to screen:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in UNIVERSE_OPTIONS
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()

    if choice is None:
        console.print("\n[red]No stock universe selected. Exiting...[/red]")
        exit(1)

    return choice


def select_max_candidates() -> int:
    """Prompt for number of stocks to deeply analyze."""
    CANDIDATE_OPTIONS = [
        ("2 stocks - Faster, lower cost (~4-10 min, ~40 LLM calls)", 2),
        ("3 stocks - Good balance (recommended, ~6-15 min, ~60 LLM calls)", 3),
        ("4 stocks - More diversified (~8-20 min, ~80 LLM calls)", 4),
        ("5 stocks - Maximum coverage (~10-25 min, ~100 LLM calls)", 5),
    ]

    choice = questionary.select(
        "How many stocks to deeply analyze?",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in CANDIDATE_OPTIONS
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()

    if choice is None:
        console.print("\n[red]No candidate count selected. Exiting...[/red]")
        exit(1)

    return choice
