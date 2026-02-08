from .state import PipelineState

# Lazy import to avoid circular dependency (runner imports agents, agents.base imports pipeline.state)
def __getattr__(name):
    if name == "TradingPipeline":
        from .runner import TradingPipeline
        return TradingPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["TradingPipeline", "PipelineState"]
