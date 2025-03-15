import os
import psycopg
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from typing import Any, List, Optional

# Load database connection details from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Load model name for execution reasoning
OPENAI_MODEL = os.getenv("OPENAI_MODEL")


@dataclass
class ExecutorDependencies:
    """Dependencies required to execute a PostgreSQL query."""
    sql_query: str  # The SQL query to be executed


class ExecutorResponse(BaseModel):
    """Defines the structured response format for the Executor Agent."""
    success: bool = Field(description="Indicates if the query executed successfully.")
    results: Optional[List[dict]] = Field(default=None, description="The query results as a list of dictionaries.")
    error_message: Optional[str] = Field(default=None, description="Error message if execution fails.")


# Define the Executor agent
executor_agent = Agent(
    OPENAI_MODEL,
    deps_type=ExecutorDependencies,  # Dependencies (SQL query)
    result_type=ExecutorResponse,  # Structured output
    system_prompt="""
    You are an AI agent responsible for executing validated PostgreSQL queries.
    
    Your task is to:
    - Execute the provided SQL query safely.
    - Return the query results as a structured JSON object.
    - If the query fails, return an error message.

    Always return a structured JSON object in this format:
    {
        "success": true or false,
        "results": "A list of dictionaries representing the query results.",
        "error_message": "If execution fails, include an error message."
    }
    
    Rules:
    - Do NOT execute queries that modify data (e.g., INSERT, UPDATE, DELETE).
    - Ensure that results are returned in a structured format.
    - If execution fails, provide a clear error message.
    """
)


def run_query(query: str) -> ExecutorResponse:
    """Executes a PostgreSQL query and returns results using psycopg3."""
    try:
        # Connect to the database
        with psycopg.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
        ) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(query)
                rows = cur.fetchall()

        # Convert results to a list of dictionaries
        return ExecutorResponse(success=True, results=rows)

    except Exception as e:
        return ExecutorResponse(success=False, error_message=str(e))


@executor_agent.tool
async def execute_query(ctx: RunContext[ExecutorDependencies], deps: ExecutorDependencies) -> ExecutorResponse:
    """
    Executes the SQL query and returns results.
    """
    return run_query(deps.sql_query)
