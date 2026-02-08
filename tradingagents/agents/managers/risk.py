"""Risk Manager (Risk Debate Judge) agent."""

from tradingagents.agents.base import BaseAgent
from tradingagents.memory.bm25 import FinancialSituationMemory
from tradingagents.pipeline.state import PipelineState


class RiskManager(BaseAgent):
    def __init__(self, client, model, memory: FinancialSituationMemory):
        super().__init__(client, model)
        self.memory = memory

    async def run(self, state: PipelineState) -> PipelineState:
        risk = state.risk_debate_state
        curr_situation = f"{state.market_report}\n\n{state.sentiment_report}\n\n{state.news_report}\n\n{state.fundamentals_report}"
        past_memories = self.memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for rec in past_memories:
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the Risk Management Judge and Debate Facilitator, your goal is to evaluate the debate between three risk analysts—Aggressive, Neutral, and Conservative—and determine the best course of action for the trader. Your decision must result in a clear recommendation: Buy, Sell, or Hold. Choose Hold only if strongly justified by specific arguments, not as a fallback when all sides seem valid. Strive for clarity and decisiveness.

Guidelines for Decision-Making:
1. **Summarize Key Arguments**: Extract the strongest points from each analyst, focusing on relevance to the context.
2. **Provide Rationale**: Support your recommendation with direct quotes and counterarguments from the debate.
3. **Refine the Trader's Plan**: Start with the trader's original plan, **{state.investment_plan}**, and adjust it based on the analysts' insights.
4. **Learn from Past Mistakes**: Use lessons from **{past_memory_str}** to address prior misjudgments and improve the decision you are making now to make sure you don't make a wrong BUY/SELL/HOLD call that loses money.

Deliverables:
- A clear and actionable recommendation: Buy, Sell, or Hold.
- Detailed reasoning anchored in the debate and past reflections.

---

**Analysts Debate History:**
{risk.history}

---

Focus on actionable insights and continuous improvement. Build on past lessons, critically evaluate all perspectives, and ensure each decision advances better outcomes."""

        response = await self._invoke(prompt)

        risk.judge_decision = response
        state.final_trade_decision = response

        return state
