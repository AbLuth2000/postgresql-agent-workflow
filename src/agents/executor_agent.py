import os
import psycopg
from typing import List, Optional
from pydantic import BaseModel, Field

# ───────────────────────────────────────────────────────────────
# Define input/output schemas
# ───────────────────────────────────────────────────────────────

class ExecutorDependencies(BaseModel):
    sql_query: str  # The SQL query to be executed

class ExecutorResponse(BaseModel):
    success: bool = Field(description="Indicates if the query executed successfully.")
    results: Optional[List[dict]] = Field(default=None, description="The query results as a list of dictionaries.")
    error_message: Optional[str] = Field(default=None, description="Error message if execution fails.")

# ───────────────────────────────────────────────────────────────
# Query Execution Logic (non-LLM)
# ───────────────────────────────────────────────────────────────

def run_query(query: str) -> ExecutorResponse:
    """
    Executes a read-only PostgreSQL query and returns results.
    """
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    try:
        # Disallow modifying queries
        disallowed = ["insert", "update", "delete", "drop", "alter"]
        if any(word in query.lower() for word in disallowed):
            return ExecutorResponse(success=False, error_message="Query modification not allowed.")

        with psycopg.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
        ) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(query)
                rows = cur.fetchall()

        return ExecutorResponse(success=True, results=rows)

    except Exception as e:
        return ExecutorResponse(success=False, error_message=str(e))

# ───────────────────────────────────────────────────────────────
# Callable function for LangGraph
# ───────────────────────────────────────────────────────────────

def execute_query(deps: ExecutorDependencies) -> ExecutorResponse:
    """
    LangChain-compatible function for executing SQL queries safely.
    """
    return run_query(deps.sql_query)
