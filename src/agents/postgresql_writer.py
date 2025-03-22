import os
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import RunnableMap

# ───────────────────────────────────────────────────────────────
# Define input and output schemas
# ───────────────────────────────────────────────────────────────

class PostgreSQLWriterDependencies(BaseModel):
    user_request: str
    database_schema: str  # The schema of the database to generate accurate queries

class PostgreSQLWriterResponse(BaseModel):
    sql_query: str = Field(description="The SQL query generated based on user request.")
    explanation: str = Field(description="A brief explanation of what the query does.")

# ───────────────────────────────────────────────────────────────
# Prompt Template
# ───────────────────────────────────────────────────────────────

prompt_template = PromptTemplate.from_template("""
You are an AI assistant responsible for generating PostgreSQL queries based on user requests.

Your task is to:
- Generate valid PostgreSQL queries that match the user's request.
- Consider the provided database schema to ensure correctness.
- Provide a brief explanation of what the query does.

Always return a structured JSON object in this format:
{{
    "sql_query": "The generated SQL query.",
    "explanation": "A brief explanation of the query's purpose."
}}

Rules:
- Always use PostgreSQL syntax.
- Ensure the query is valid based on the given database schema.
- If the user request is unclear, generate the most logical query.
- DO NOT execute the query; just return it.

User Request: {user_request}
Database Schema: {database_schema}
""")

# ───────────────────────────────────────────────────────────────
# Initialize LLM
# ───────────────────────────────────────────────────────────────

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

# ───────────────────────────────────────────────────────────────
# Create Runnable Agent
# ───────────────────────────────────────────────────────────────

postgresql_writer_agent = (
    prompt_template
    | llm
    | (lambda x: PostgreSQLWriterResponse.model_validate_json(x))
)

# ───────────────────────────────────────────────────────────────
# Callable function (for LangGraph or other orchestration)
# ───────────────────────────────────────────────────────────────

def generate_query(deps: PostgreSQLWriterDependencies) -> PostgreSQLWriterResponse:
    """
    Generates a PostgreSQL query and explanation using LangChain pipeline.
    """
    return postgresql_writer_agent.invoke({
        "user_request": deps.user_request,
        "database_schema": deps.database_schema
    })
