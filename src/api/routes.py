from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.core.workflow import workflow, AgentState  # Import LangGraph workflow

router = APIRouter()

class UserRequest(BaseModel):
    user_input: str

class AgentResponse(BaseModel):
    next_agent: str
    response: dict

@router.post("/process", response_model=AgentResponse)
async def process_request(request: UserRequest):
    # Initialize the state with user input
    state = AgentState(user_input=request.input_text, next_agent="orchestrator", response={})
    # Execute the workflow
    try:
        final_state = await workflow.run(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Extract the response from the final state
    output_text = final_state.response.get('result', 'No response generated.')
    return AgentResponse(output_text=output_text)
