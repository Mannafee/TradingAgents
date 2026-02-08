"""Trader agent."""

from tradingagents.agents.base import BaseAgent
from tradingagents.memory.bm25 import FinancialSituationMemory
from tradingagents.pipeline.state import PipelineState


SYSTEM_PROMPT = """You are a trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold. End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation. Do not forget to utilize lessons from past decisions to learn from your mistakes. Here is some reflections from similar situations you traded in and the lessons learned: {past_memory_str}"""


class Trader(BaseAgent):
    def __init__(self, client, model, memory: FinancialSituationMemory):
        super().__init__(client, model)
        self.memory = memory

    async def run(self, state: PipelineState) -> PipelineState:
        curr_situation = f"{state.market_report}\n\n{state.sentiment_report}\n\n{state.news_report}\n\n{state.fundamentals_report}"
        past_memories = self.memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for rec in past_memories:
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        system = SYSTEM_PROMPT.format(past_memory_str=past_memory_str)

        prompt = f"""Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {state.company_of_interest}. This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.

Proposed Investment Plan: {state.investment_plan}

Leverage these insights to make an informed and strategic decision."""

        response = await self._invoke(prompt, system=system)
        state.trader_investment_plan = response

        return state
