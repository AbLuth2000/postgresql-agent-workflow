from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.core.workflow import workflow  # Import LangGraph workflow

router = APIRouter()

class UserRequest(BaseModel):
    user_input: str

class AgentResponse(BaseModel):
    next_agent: str
    response: dict

@router.post("/query", response_model=AgentResponse)
def process_request(request: UserRequest):
    """
    Processes a user query and returns the response from the agentic workflow.
    """
    try:
        state = {"user_input": request.user_input}
        for result in workflow.stream(state):
            if "next_agent" in result:
                return AgentResponse(next_agent=result["next_agent"], response=result)
        raise HTTPException(status_code=500, detail="Workflow did not return a valid response.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
def health_check():
    return {"status": "ok", "message": "Agentic Workflow API is running."}
