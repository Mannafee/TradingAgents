import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "data/data_cache",
    ),
    # LLM settings (Codex-only)
    "deep_think_llm": "gpt-5.2-codex",
    "quick_think_llm": "gpt-5.1-codex-mini",
    "openai_reasoning_effort": None,
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    # Portfolio advisor settings
    "portfolio_default_universe": "sp500_top50",
    "portfolio_screening_period": "1mo",
}


# Module-level config state (replaces dataflows/config.py)
_config = None


def get_config() -> dict:
    global _config
    if _config is None:
        _config = DEFAULT_CONFIG.copy()
    return _config.copy()


def set_config(config: dict):
    global _config
    if _config is None:
        _config = DEFAULT_CONFIG.copy()
    _config.update(config)
