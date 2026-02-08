"""Bull Researcher agent."""

from tradingagents.agents.base import BaseAgent
from tradingagents.memory.bm25 import FinancialSituationMemory
from tradingagents.pipeline.state import PipelineState


class BullResearcher(BaseAgent):
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

        prompt = f"""You are a Bull Analyst advocating for investing in the stock. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

Resources available:
Market research report: {state.market_report}
Social media sentiment report: {state.sentiment_report}
Latest world affairs news: {state.news_report}
Company fundamentals report: {state.fundamentals_report}
Conversation history of the debate: {debate.history}
Last bear argument: {debate.current_response}
Reflections from similar situations and lessons learned: {past_memory_str}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position. You must also address reflections and learn from lessons and mistakes you made in the past.
"""

        response = await self._invoke(prompt)
        argument = f"Bull Analyst: {response}"

        debate.history = debate.history + "\n" + argument
        debate.bull_history = debate.bull_history + "\n" + argument
        debate.current_response = argument
        debate.count += 1

        return state
