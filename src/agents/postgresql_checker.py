import os
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from typing import Optional

# Load model name from environment variables
OPENAI_MODEL = os.getenv("OPENAI_MODEL")


@dataclass
class PostgreSQLCheckerDependencies:
    """Dependencies required for validating a PostgreSQL query."""
    user_request: str  # The original user request
    sql_query: str  # The generated SQL query
    database_schema: str  # The database schema for validation


class PostgreSQLCheckerResponse(BaseModel):
    """Defines the structured response format for the PostgreSQL Checker."""
    is_valid: bool = Field(description="Whether the query correctly fulfills the user request.")
    reason: str = Field(description="Explanation of why the query is valid or invalid.")
    suggested_fix: Optional[str] = Field(default=None, description="A suggested fix if the query is invalid.")
    expected_output: Optional[str] = Field(default=None, description="An example of what the query output should look like.")


# Define the PostgreSQL Checker agent
PostgreSQLCheckerAgent = Agent(
    OPENAI_MODEL,
    deps_type=PostgreSQLCheckerDependencies,  # Dependencies (query, request, schema)
    result_type=PostgreSQLCheckerResponse,  # Structured output
    system_prompt="""
    You are an AI assistant responsible for validating PostgreSQL queries.
    
    Your task is to:
    - Determine whether the provided SQL query correctly fulfills the user's request.
    - Use the provided database schema to verify that the query will run correctly.
    - Explain why the query is valid or invalid.
    - If the query is invalid, suggest an improved version.
    - Attempt to generate an **expected sample output** based on the schema.

    Always return a structured JSON object in this format:
    {
        "is_valid": true or false,
        "reason": "Explanation of why the query is valid/invalid.",
        "suggested_fix": "If invalid, a corrected version of the SQL query.",
        "expected_output": "A JSON representation of what the query output might look like."
    }
    
    Rules:
    - Ensure the query fully satisfies the **intent** of the user request.
    - Ensure the SQL syntax follows PostgreSQL standards.
    - If the query is invalid, provide a **corrected version**.
    - The expected output should be an **accurate guess** of the query results.
    """
)


@PostgreSQLCheckerAgent.tool
async def validate_query(ctx: RunContext[PostgreSQLCheckerDependencies], deps: PostgreSQLCheckerDependencies) -> PostgreSQLCheckerResponse:
    """
    Validates the SQL query by checking if it fulfills the user's request and conforms to PostgreSQL syntax.
    """
    return await PostgreSQLCheckerAgent.run(deps.user_request, deps=deps)
