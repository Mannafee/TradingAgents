"""Post-decision reflection - generates lessons learned and updates memory."""

from tradingagents.client.codex import CodexClient
from tradingagents.client.types import Message
from tradingagents.memory.bm25 import FinancialSituationMemory
from tradingagents.pipeline.state import PipelineState


REFLECTION_PROMPT = """
You are an expert financial analyst tasked with reviewing trading decisions/analysis and providing a comprehensive, step-by-step analysis.
Your goal is to deliver detailed insights into investment decisions and highlight opportunities for improvement, adhering strictly to the following guidelines:

1. Reasoning:
   - For each trading decision, determine whether it was correct or incorrect. A correct decision results in an increase in returns, while an incorrect decision does the opposite.
   - Analyze the contributing factors to each success or mistake. Consider:
     - Market intelligence.
     - Technical indicators.
     - Technical signals.
     - Price movement analysis.
     - Overall market data analysis
     - News analysis.
     - Social media and sentiment analysis.
     - Fundamental data analysis.
     - Weight the importance of each factor in the decision-making process.

2. Improvement:
   - For any incorrect decisions, propose revisions to maximize returns.
   - Provide a detailed list of corrective actions or improvements, including specific recommendations (e.g., changing a decision from HOLD to BUY on a particular date).

3. Summary:
   - Summarize the lessons learned from the successes and mistakes.
   - Highlight how these lessons can be adapted for future trading scenarios and draw connections between similar situations to apply the knowledge gained.

4. Query:
   - Extract key insights from the summary into a concise sentence of no more than 1000 tokens.
   - Ensure the condensed sentence captures the essence of the lessons and reasoning for easy reference.

Adhere strictly to these instructions, and ensure your output is detailed, accurate, and actionable. You will also be given objective descriptions of the market from a price movements, technical indicator, news, and sentiment perspective to provide more context for your analysis.
"""


class Reflector:
    """Generates reflections on decisions and updates BM25 memory."""

    def __init__(self, client: CodexClient, model: str):
        self.client = client
        self.model = model

    def _extract_situation(self, state: PipelineState) -> str:
        return f"{state.market_report}\n\n{state.sentiment_report}\n\n{state.news_report}\n\n{state.fundamentals_report}"

    async def _reflect(self, report: str, situation: str, returns_losses) -> str:
        messages = [
            Message(role="system", content=REFLECTION_PROMPT),
            Message(
                role="user",
                content=f"Returns: {returns_losses}\n\nAnalysis/Decision: {report}\n\nObjective Market Reports for Reference: {situation}",
            ),
        ]
        response = await self.client.complete(messages, model=self.model)
        return response.text

    async def reflect_and_remember(
        self,
        state: PipelineState,
        returns_losses,
        bull_memory: FinancialSituationMemory,
        bear_memory: FinancialSituationMemory,
        trader_memory: FinancialSituationMemory,
        invest_judge_memory: FinancialSituationMemory,
        risk_manager_memory: FinancialSituationMemory,
    ):
        """Reflect on all components and update their memories."""
        situation = self._extract_situation(state)

        # Bull researcher
        result = await self._reflect(
            state.investment_debate_state.bull_history, situation, returns_losses
        )
        bull_memory.add_situations([(situation, result)])

        # Bear researcher
        result = await self._reflect(
            state.investment_debate_state.bear_history, situation, returns_losses
        )
        bear_memory.add_situations([(situation, result)])

        # Trader
        result = await self._reflect(
            state.trader_investment_plan, situation, returns_losses
        )
        trader_memory.add_situations([(situation, result)])

        # Investment judge
        result = await self._reflect(
            state.investment_debate_state.judge_decision, situation, returns_losses
        )
        invest_judge_memory.add_situations([(situation, result)])

        # Risk manager
        result = await self._reflect(
            state.risk_debate_state.judge_decision, situation, returns_losses
        )
        risk_manager_memory.add_situations([(situation, result)])
