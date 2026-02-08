"""Research Manager (Investment Debate Judge) agent."""

from tradingagents.agents.base import BaseAgent
from tradingagents.memory.bm25 import FinancialSituationMemory
from tradingagents.pipeline.state import PipelineState


class ResearchManager(BaseAgent):
    def __init__(self, client, model, memory: FinancialSituationMemory):
        super().__init__(client, model)
        self.memory = memory

    async def run(self, state: PipelineState) -> PipelineState:
        debate = state.investment_debate_state
        curr_situation = f"{state.market_report}\n\n{state.sentiment_report}\n\n{state.news_report}\n\n{state.fundamentals_report}"
        past_memories = self.memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for rec in past_memories:
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the portfolio manager and debate facilitator, your role is to critically evaluate this round of debate and make a definitive decision: align with the bear analyst, the bull analyst, or choose Hold only if it is strongly justified based on the arguments presented.

Summarize the key points from both sides concisely, focusing on the most compelling evidence or reasoning. Your recommendation—Buy, Sell, or Hold—must be clear and actionable. Avoid defaulting to Hold simply because both sides have valid points; commit to a stance grounded in the debate's strongest arguments.

Additionally, develop a detailed investment plan for the trader. This should include:

Your Recommendation: A decisive stance supported by the most convincing arguments.
Rationale: An explanation of why these arguments lead to your conclusion.
Strategic Actions: Concrete steps for implementing the recommendation.
Take into account your past mistakes on similar situations. Use these insights to refine your decision-making and ensure you are learning and improving. Present your analysis conversationally, as if speaking naturally, without special formatting.

Here are your past reflections on mistakes:
\"{past_memory_str}\"

Here is the debate:
Debate History:
{debate.history}"""

        response = await self._invoke(prompt)

        debate.judge_decision = response
        debate.current_response = response
        state.investment_plan = response

        return state
