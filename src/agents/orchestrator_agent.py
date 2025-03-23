from dotenv import load_dotenv
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

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
You are a helpful user assistant. You handle all communication, decisions, and follow-up questions in a workflow.
Always reply clearly, and keep track of the conversation history.

You can do the following:
- Decide what the user is asking for
- Ask follow-up questions if the request is unclear
- Talk to an internal analyst agent to get insights
- Maintain multi-turn conversations with the analyst

If the user is continuing an ongoing conversation with the analyst, keep routing to the analyst agent unless the user changes the topic.

Return one of:
- "analyst"
- "postgresql_writer"
- "postgresql_checker"
- "executor"
- "follow_up"
- "complete"

User Input: {input}
""")

# ───────────────────────────────────────────────────────────────
# Initialize the LLM
# ───────────────────────────────────────────────────────────────

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0
)

# ───────────────────────────────────────────────────────────────
# Create runnable pipeline
# ───────────────────────────────────────────────────────────────

orchestrator_agent = (
    prompt_template
    | llm
    | (lambda x: OrchestratorResponse.model_validate_json(x.content))
)

# ───────────────────────────────────────────────────────────────
# Callable function to invoke the orchestrator
# ───────────────────────────────────────────────────────────────

def route_request(user_input: str, message_history: Optional[List[Dict[str, str]]] = None) -> OrchestratorResponse:
    """
    Analyzes the user's request and routes it to the appropriate agent.
    """

    if message_history is None:
        message_history = []

    # Add the latest user input to the message history
    updated_history = message_history + [{"role": "user", "content": user_input}]

    # Prepend the system prompt (should always come first)
    messages = [{"role": "system", "content": prompt_template.format(input=user_input)}] + updated_history

    # Use LangChain ChatOpenAI directly with messages
    response = llm.invoke(messages)

    # Parse model response (assumed to be structured JSON)
    parsed = OrchestratorResponse.model_validate_json(response.content)

    return parsed

