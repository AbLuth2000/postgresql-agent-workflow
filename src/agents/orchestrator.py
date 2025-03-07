import os
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from pydantic_ai import Agent, OpenAIChatCompletion

# Load model name and API key from environment variables
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class OrchestratorResponse(BaseModel):
    """
    Defines the structured response format for the orchestrator.
    """
    decision: str = Field(
        description="The next step for the workflow. Must be one of: 'analyst', 'postgresql_writer', 'postgresql_checker', 'executor', 'follow_up', or 'complete'."
    )
    follow_up_question: Optional[str] = Field(
        default=None,
        description="A question to ask the user if more information is needed (only used if decision is 'follow_up').",
    )

class OrchestratorAgent:
    """
    Orchestrator Agent:
    - Uses an LLM to determine the next step based on user input.
    - Ensures structured responses using PydanticAI.
    """
    def __init__(self, model_name: str = OPENAI_MODEL, temperature: float = 0.2):
        """Initialize the orchestrator with an LLM model."""
        self.llm = OpenAIChatCompletion(model=model_name, api_key=OPENAI_API_KEY, temperature=temperature)
        self.agent = Agent(llm=self.llm, output_model=OrchestratorResponse)

    def route_request(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses the LLM to determine the next step in the workflow.
        Returns a structured response specifying the next agent or follow-up action.
        """
        user_input = state.get("user_input", "")

        # Define system instruction for structured decision-making
        system_prompt = """
            You are an orchestrator in an AI workflow. Your job is to decide the next step based on the user's request.

            You must return a structured response with a decision from the following options:

            - "analyst": If the user wants insights or explanations about the database.
            - "postgresql_writer": If the user wants a PostgreSQL query to be written.
            - "postgresql_checker": If a query needs to be validated before execution.
            - "executor": If the user provides a query and wants it executed.
            - "follow_up": If the user's request is unclear and you need more details. In this case, include a relevant follow-up question.
            - "complete": If the user's request has been fully resolved and no further action is needed.

            Always return a JSON object in this format:
            {
                "decision": "AGENT_NAME or 'follow_up' or 'complete'",
                "follow_up_question": "Only included if decision is 'follow_up'. Otherwise, set to null."
            }
        """

        # Use the agent to process the input and obtain a structured response
        response = self.agent.run(system_prompt=system_prompt, input=user_input)

        return response.dict()  # Convert structured response to a dictionary
