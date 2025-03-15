import os
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic.dataclasses import dataclass
from typing import Optional

# Load model name for analysis
OPENAI_MODEL = os.getenv("OPENAI_MODEL")


class OrchestratorResponse(BaseModel):
    """Defines the structured response format for the orchestrator."""
    decision: str = Field(
        description="The next step in the workflow. Must be one of: 'analyst', 'postgresql_writer', 'postgresql_checker', 'executor', 'follow_up', or 'complete'."
    )
    follow_up_question: Optional[str] = Field(
        default=None,
        description="A question to ask the user if more information is needed (only used if decision is 'follow_up').",
    )


# Define the orchestrator agent
OrchestratorAgent = Agent(
    OPENAI_MODEL,
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


@OrchestratorAgent.tool
async def route_request(ctx: RunContext[str]) -> OrchestratorResponse:
    """
    Routes the user's request to the appropriate agent.
    """
    return await OrchestratorAgent.invoke(ctx.deps)
