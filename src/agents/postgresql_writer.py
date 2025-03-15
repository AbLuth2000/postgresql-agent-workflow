import os
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

# Load model name from environment variables
OPENAI_MODEL = os.getenv("OPENAI_MODEL")


@dataclass
class PostgreSQLWriterDependencies:
    """Dependencies required for generating a PostgreSQL query."""
    user_request: str
    database_schema: str  # The schema of the database to generate accurate queries


class PostgreSQLWriterResponse(BaseModel):
    """Defines the structured response format for the PostgreSQL Writer."""
    sql_query: str = Field(description="The SQL query generated based on user request.")
    explanation: str = Field(description="A brief explanation of what the query does.")


# Define the PostgreSQL Writer agent
postgresql_writer_agent = Agent(
    OPENAI_MODEL,
    deps_type=PostgreSQLWriterDependencies,  # Dependencies (user request & schema)
    result_type=PostgreSQLWriterResponse,  # Structured output
    system_prompt="""
    You are an AI assistant responsible for generating PostgreSQL queries based on user requests.
    
    Your task is to:
    - Generate **valid** PostgreSQL queries that match the user's request.
    - Consider the provided **database schema** to ensure correctness.
    - Provide a **brief explanation** of what the query does.
    
    Always return a structured JSON object in this format:
    {
        "sql_query": "The generated SQL query.",
        "explanation": "A brief explanation of the query's purpose."
    }
    
    Rules:
    - Always use PostgreSQL syntax.
    - Ensure the query is valid based on the given database schema.
    - If the user request is unclear, generate the most **logical** query.
    - DO NOT execute the query; just return it.
    """
)


@postgresql_writer_agent.tool
async def generate_query(ctx: RunContext[PostgreSQLWriterDependencies], deps: PostgreSQLWriterDependencies) -> PostgreSQLWriterResponse:
    """
    Generates a PostgreSQL query based on the user's request and database schema.
    """
    return await postgresql_writer_agent.run(deps.user_request, deps=deps)
