import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import RunnableMap

# ───────────────────────────────────────────────────────────────
# Define input dependencies and output schema
# ───────────────────────────────────────────────────────────────

class AnalystDependencies(BaseModel):
    user_request: str  # The original user request
    query_results: List[dict]  # The query results from the Executor Agent

class AnalystResponse(BaseModel):
    insights: str = Field(description="A human-readable summary of the data.")
    key_findings: List[str] = Field(description="Important insights or trends detected in the data.")
    next_steps: Optional[str] = Field(default=None, description="Suggested actions based on the data.")

# ───────────────────────────────────────────────────────────────
# Define system prompt template
# ───────────────────────────────────────────────────────────────

prompt_template = PromptTemplate.from_template("""
You are an AI analyst responsible for interpreting query results and providing insights.

Your task is to:
- Analyze the query results and generate meaningful insights.
- Summarize key findings in a human-readable format.
- Identify trends, anomalies, and patterns in the data.
- Suggest next steps if applicable.

Always return a structured JSON object in this format:
{{
    "insights": "A summary of the query results.",
    "key_findings": ["A list of important insights."],
    "next_steps": "Optional recommended actions."
}}

Rules:
- Keep explanations clear and concise.
- If data is missing or unusual, highlight anomalies.
- Suggest next steps based on patterns in the data.

User Request: {user_request}
Query Results: {query_results}
""")

# ───────────────────────────────────────────────────────────────
# Initialize the language model
# ───────────────────────────────────────────────────────────────

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

# ───────────────────────────────────────────────────────────────
# Create the analyst agent as a Runnable pipeline
# ───────────────────────────────────────────────────────────────

analyst_agent = prompt_template | llm | (lambda x: AnalystResponse.model_validate_json(x))

# ───────────────────────────────────────────────────────────────
# Wrap in a callable function for LangGraph or direct use
# ───────────────────────────────────────────────────────────────

def analyze_request(deps: AnalystDependencies) -> AnalystResponse:
    """
    Analyzes query results and provides insights using LangChain agent.
    """
    return analyst_agent.invoke({
        "user_request": deps.user_request,
        "query_results": deps.query_results
    })
