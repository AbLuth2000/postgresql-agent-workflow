from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import json

# ───────────────────────────────────────────────────────────────
# Define input dependencies and output schema
# ───────────────────────────────────────────────────────────────

class AnalystDependencies(BaseModel):
    user_request: str

class AnalystResponse(BaseModel):
    insights: str = Field(description="A human-readable summary of the data.")
    key_findings: List[str] = Field(description="Important insights or trends detected in the data.")
    next_steps: Optional[str] = Field(default=None, description="Suggested actions based on the data.")

# ───────────────────────────────────────────────────────────────
# Prompt template
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
""")

# ───────────────────────────────────────────────────────────────
# LLM config
# ───────────────────────────────────────────────────────────────

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ───────────────────────────────────────────────────────────────
# Output parser
# ───────────────────────────────────────────────────────────────

def parse_analyst_response(ai_message) -> AnalystResponse:
    """
    Parse AIMessage content into a Pydantic response object.
    """
    try:
        return AnalystResponse.model_validate_json(ai_message.content)
    except Exception:
        return AnalystResponse(**json.loads(ai_message.content))

# ───────────────────────────────────────────────────────────────
# Chain pipeline
# ───────────────────────────────────────────────────────────────

analyst_agent = prompt_template | llm | parse_analyst_response

# ───────────────────────────────────────────────────────────────
# Callable function (can be used with LangGraph or directly)
# ───────────────────────────────────────────────────────────────

def analyze_request(deps: AnalystDependencies) -> AnalystResponse:
    """
    Analyzes raw query results based on a user request and returns structured insights.
    """
    return analyst_agent.invoke({
        "user_request": deps.user_request
    })
