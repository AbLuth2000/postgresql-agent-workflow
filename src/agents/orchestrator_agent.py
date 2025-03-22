import os
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# ───────────────────────────────────────────────────────────────
# Define input and output schemas
# ───────────────────────────────────────────────────────────────

class OrchestratorResponse(BaseModel):
    decision: str = Field(
        description="The next step in the workflow. Must be one of: 'analyst', 'postgresql_writer', 'postgresql_checker', 'executor', 'follow_up', or 'complete'."
    )
    follow_up_question: Optional[str] = Field(
        default=None,
        description="A question to ask the user if more information is needed (only used if decision is 'follow_up').",
    )

# ───────────────────────────────────────────────────────────────
# Prompt template
# ───────────────────────────────────────────────────────────────

prompt_template = PromptTemplate.from_template("""
You are an orchestrator in an AI workflow. Your job is to decide the next step based on the user's request.

You must return a structured response with a decision from the following options:
- 'analyst': If the user wants insights or explanations about the database.
- 'postgresql_writer': If the user wants a PostgreSQL query to be written.
- 'postgresql_checker': If a query needs to be validated before execution.
- 'executor': If the user provides a query and wants it executed.
- 'follow_up': If the user's request is unclear and you need more details. In this case, include a relevant follow-up question.
- 'complete': If the user's request has been fully resolved and no further action is needed.

Always return a JSON object in this format:
{{
    "decision": "AGENT_NAME or follow_up or complete",
    "follow_up_question": "Only included if decision is follow_up. Otherwise, set to null."
}}

User Input: {input}
""")

# ───────────────────────────────────────────────────────────────
# Initialize the LLM
# ───────────────────────────────────────────────────────────────

load_dotenv()

llm = OpenAI(
    model="gpt-4o-mini", 
    temperature=0
)

# ───────────────────────────────────────────────────────────────
# Create runnable pipeline
# ───────────────────────────────────────────────────────────────

orchestrator_agent = (
    prompt_template
    | llm
    | (lambda x: OrchestratorResponse.model_validate_json(x))
)

# ───────────────────────────────────────────────────────────────
# Callable function to invoke the orchestrator
# ───────────────────────────────────────────────────────────────

def route_request(user_input: str) -> OrchestratorResponse:
    """
    Analyzes the user's request and routes it to the appropriate agent.
    """
    return orchestrator_agent.invoke({"input": user_input})
