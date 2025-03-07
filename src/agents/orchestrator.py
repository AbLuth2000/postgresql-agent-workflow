import os
from pydantic_ai import Agent, RunContext
from pydantic.dataclasses import dataclass
from typing import Optional


@dataclass
class OrchestratorResponse:
    """
    Defines the structured response format for the orchestrator.
    """
    decision: str
    follow_up_question: Optional[str] = None


# Define the orchestrator agent
orchestrator_agent = Agent(
    OPENAI_MODEL = os.getenv("OPENAI_MODEL"),
    deps_type=str,  # User input as dependency
    result_type=OrchestratorResponse,  # Enforce structured output
    system_prompt="""
        You are an orchestrator in an AI workflow. Your job is to decide the next step based on the user's request.

        You must return a structured response with a decision from the following options:
        - 'analyst': If the user wants insights or explanations about the database.
        - 'postgresql_writer': If the user wants a PostgreSQL query to be written.
        - 'postgresql_checker': If a query needs to be validated before execution.
        - 'executor': If the user provides a query and wants it executed.
        - 'follow_up': If the user's request is unclear and you need more details. In this case, include a relevant follow-up question.
        - 'complete': If the user's request has been fully resolved and no further action is needed.

        Always return a JSON object in this format:
        {
            "decision": "AGENT_NAME or follow_up or complete",
            "follow_up_question": "Only included if decision is follow_up. Otherwise, set to null."
        }
    """
)


@orchestrator_agent.tool
async def route_request(ctx: RunContext[str], user_input: str) -> OrchestratorResponse:
    """
    Routes the user's request to the appropriate agent.
    """
    return await orchestrator_agent.run(user_input, deps=user_input)
