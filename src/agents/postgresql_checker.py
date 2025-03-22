import os
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import RunnableMap

# ───────────────────────────────────────────────────────────────
# Define input dependencies and output schema
# ───────────────────────────────────────────────────────────────

class PostgreSQLCheckerDependencies(BaseModel):
    user_request: str  # The original user request
    sql_query: str  # The generated SQL query
    database_schema: str  # The database schema for validation

class PostgreSQLCheckerResponse(BaseModel):
    is_valid: bool = Field(description="Whether the query correctly fulfills the user request.")
    reason: str = Field(description="Explanation of why the query is valid or invalid.")
    suggested_fix: Optional[str] = Field(default=None, description="A suggested fix if the query is invalid.")
    expected_output: Optional[str] = Field(default=None, description="An example of what the query output should look like.")

# ───────────────────────────────────────────────────────────────
# System prompt template
# ───────────────────────────────────────────────────────────────

prompt_template = PromptTemplate.from_template("""
You are an AI assistant responsible for validating PostgreSQL queries.

Your task is to:
- Determine whether the provided SQL query correctly fulfills the user's request.
- Use the provided database schema to verify that the query will run correctly.
- Explain why the query is valid or invalid.
- If the query is invalid, suggest an improved version.
- Attempt to generate an expected sample output based on the schema.

Always return a structured JSON object in this format:
{{
    "is_valid": true or false,
    "reason": "Explanation of why the query is valid/invalid.",
    "suggested_fix": "If invalid, a corrected version of the SQL query.",
    "expected_output": "A JSON representation of what the query output might look like."
}}

Rules:
- Ensure the query fully satisfies the intent of the user request.
- Ensure the SQL syntax follows PostgreSQL standards.
- If the query is invalid, provide a corrected version.
- The expected output should be an accurate guess of the query results.

User Request: {user_request}
SQL Query: {sql_query}
Database Schema: {database_schema}
""")

# ───────────────────────────────────────────────────────────────
# Initialize the LLM
# ───────────────────────────────────────────────────────────────

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

# ───────────────────────────────────────────────────────────────
# Create the runnable pipeline
# ───────────────────────────────────────────────────────────────

postgresql_checker_agent = (
    prompt_template 
    | llm 
    | (lambda x: PostgreSQLCheckerResponse.model_validate_json(x))
)

# ───────────────────────────────────────────────────────────────
# Callable function (e.g. for LangGraph integration)
# ───────────────────────────────────────────────────────────────

def validate_query(deps: PostgreSQLCheckerDependencies) -> PostgreSQLCheckerResponse:
    """
    Validates the SQL query against the user's request and schema using LangChain.
    """
    return postgresql_checker_agent.invoke({
        "user_request": deps.user_request,
        "sql_query": deps.sql_query,
        "database_schema": deps.database_schema
    })
