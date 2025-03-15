import os
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from typing import Any, List, Optional

# Load model name for analysis
OPENAI_MODEL = os.getenv("OPENAI_MODEL")


@dataclass
class AnalystDependencies:
    """Dependencies required for analyzing query results."""
    user_request: str  # The original user request
    query_results: List[dict]  # The query results from the Executor Agent


class AnalystResponse(BaseModel):
    """Defines the structured response format for the Analyst Agent."""
    insights: str = Field(description="A human-readable summary of the data.")
    key_findings: List[str] = Field(description="Important insights or trends detected in the data.")
    next_steps: Optional[str] = Field(default=None, description="Suggested actions based on the data.")


# Define the Analyst agent
analyst_agent = Agent(
    OPENAI_MODEL,
    deps_type=AnalystDependencies,  # Dependencies (query results)
    result_type=AnalystResponse,  # Structured output
    system_prompt="""
    You are an AI analyst responsible for interpreting query results and providing insights.
    
    Your task is to:
    - Analyze the query results and generate meaningful insights.
    - Summarize key findings in a human-readable format.
    - Identify trends, anomalies, and patterns in the data.
    - Suggest next steps if applicable.

    Always return a structured JSON object in this format:
    {
        "insights": "A summary of the query results.",
        "key_findings": ["A list of important insights."],
        "next_steps": "Optional recommended actions."
    }
    
    Rules:
    - Keep explanations clear and concise.
    - If data is missing or unusual, highlight anomalies.
    - Suggest next steps based on patterns in the data.
    """
)


@analyst_agent.tool
async def analyze_request(ctx: RunContext[AnalystDependencies], deps: AnalystDependencies) -> AnalystResponse:
    """
    Analyzes query results and provides insights.
    """
    return await analyst_agent.run(deps.user_request, deps=deps)
