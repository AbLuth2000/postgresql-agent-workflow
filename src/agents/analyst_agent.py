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
You are an AI analyst assistant. The user is asking for data insights, explanations, or breakdowns.

Your job is to:
- Understand their request
- Provide meaningful analysis
- Identify trends, patterns, or summaries based on their intent
- Propose next steps if helpful

Always return a JSON object in this format:
{{
  "insights": "A short summary of what the user is asking about.",
  "key_findings": ["A list of notable insights."],
  "next_steps": "Recommended actions or follow-up questions (optional)."
}}

User Request: {user_request}
""")

# ───────────────────────────────────────────────────────────────
# LLM config
# ───────────────────────────────────────────────────────────────

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
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
    Provides user structured insights.
    """
    result = analyst_agent.invoke({"user_request": deps.user_request})

    print("\n[Analyst Agent Output]")
    print(result)

    return result